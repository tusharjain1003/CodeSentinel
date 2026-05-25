from openai import AsyncOpenAI

from agents.base import ReviewComment
from config import settings

client = AsyncOpenAI(api_key=settings.openai_api_key)


async def judge_comment_quality(diff: str, comment: ReviewComment, ground_truth: str) -> int:
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": f"""Score this AI code review comment from 0 to 3.

Code diff:
```diff
{diff}
```

AI comment: {comment.message}
AI suggestion: {comment.suggestion or "None"}
Ground truth issue: {ground_truth}

Respond with just the integer score.""",
            }
        ],
        max_tokens=3,
    )
    return int((response.choices[0].message.content or "0").strip())
