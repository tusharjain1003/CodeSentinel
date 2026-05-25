import asyncio
from pathlib import Path

import asyncpg

from config import settings


async def setup() -> None:
    sql = Path("db/schema.sql").read_text(encoding="utf-8")
    connection = await asyncpg.connect(settings.database_url)
    try:
        await connection.execute(sql)
    finally:
        await connection.close()


if __name__ == "__main__":
    asyncio.run(setup())
