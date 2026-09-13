import asyncio
import json
import sys

from fastmcp import Client

from config import API_HOST, API_PORT

URL = "http://%s:%s/mcp" % (API_HOST, API_PORT)
RECORD_IDS = ["NYK-1001", "NYK-1010"]


async def call_two_orders(url=URL):
    rows = []
    async with Client(url) as client:
        for rid in RECORD_IDS:
            result = await client.call_tool("check_order_status", {"record_id": rid})
            payload = result.data if hasattr(result, "data") else result
            if hasattr(payload, "model_dump"):
                payload = payload.model_dump()
            row = {"record_id": rid, "mcp_response": payload}
            rows.append(row)
            print(json.dumps(row, indent=2, default=str))
    return rows


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else URL
    asyncio.run(call_two_orders(url))


if __name__ == "__main__":
    main()
