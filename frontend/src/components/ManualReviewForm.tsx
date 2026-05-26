import { Play } from "lucide-react";
import { FormEvent, useState } from "react";

export function ManualReviewForm() {
  const [status, setStatus] = useState("");

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    setStatus("Submitting");
    const response = await fetch("/api/review", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pr_url: form.get("pr_url"),
        repo: form.get("repo"),
        pr_number: Number(form.get("pr_number")),
      })
    });
    setStatus(response.ok ? "Accepted" : "Failed");
  }

  return (
    <section className="panel">
      <div className="panelHeader">
        <h2>Manual Review</h2>
      </div>
      <form onSubmit={submit} className="reviewForm">
        <label>
          <span>Repository</span>
          <input name="repo" placeholder="owner/repo" required />
        </label>
        <label>
          <span>PR Number</span>
          <input name="pr_number" type="number" min="1" required />
        </label>
        <label>
          <span>PR URL</span>
          <input name="pr_url" type="url" required />
        </label>
        <button type="submit">
          <Play size={16} />
          Run Review
        </button>
      </form>
      {status ? <div className="status">{status}</div> : null}
    </section>
  );
}
