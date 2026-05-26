import json

import modal

app = modal.App("codesentinel-inference-test")

inference_image = (
    modal.Image.debian_slim()
    .pip_install(
        "transformers>=4.43",
        "accelerate>=0.33",
        "torch>=2.3",
    )
)

checkpoints_vol = modal.Volume.from_name("codesentinel-checkpoints", create_if_missing=True)


@app.function(
    image=inference_image,
    gpu="A10G",
    timeout=300,
    volumes={"/checkpoints": checkpoints_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def test_inference() -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = "/checkpoints/codesentinel-merged"
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.config.use_cache = True

    system_prompt = (
        "You are an expert code reviewer. Analyze the provided code diff and identify "
        "issues. Respond only with a valid JSON object matching the ReviewComment schema."
    )
    user_prompt = """Review this code change in `src/auth.py`:

```diff
@@ -1,7 +1,7 @@
 def get_user(username):
-    query = "SELECT * FROM users WHERE name='" + username + "'"
+    query = f"SELECT * FROM users WHERE name='{username}'"
     return db.execute(query)

 def lookup(email):
-    query = "SELECT * FROM users WHERE email='" + email + "'"
+    query = f"SELECT * FROM users WHERE email='{email}'"
     return db.execute(query)
```

Identify any issues. Respond with a JSON ReviewComment object."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        inputs,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    print(f"Raw response:\n{response}")

    try:
        parsed = json.loads(response)
        print(f"\nParsed JSON:\n{json.dumps(parsed, indent=2)}")
        return json.dumps({"status": "ok", "response": parsed})
    except json.JSONDecodeError as e:
        print(f"\nFailed to parse JSON: {e}")
        return json.dumps({"status": "parse_failed", "raw": response})


if __name__ == "__main__":
    result = test_inference.remote()
    print(f"\nResult: {result}")
