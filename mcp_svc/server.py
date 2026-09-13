from fastmcp import FastMCP

from agent.tools import check_order_status as lookup_order
from config import API_HOST, API_PORT

mcp = FastMCP("Nykaa Order Tools")


@mcp.tool()
def check_order_status(record_id: str) -> dict:
    """Look up a Nykaa order by record_id and return status, order_value_inr and a 0-1 escalation_score.

    record_id looks like NYK-1001. Unknown ids come back with found=false.
    """
    return lookup_order(record_id)


mcp_app = mcp.http_app(path="/")


def run_standalone():
    mcp.run(transport="http", host=API_HOST, port=API_PORT, path="/mcp")


if __name__ == "__main__":
    run_standalone()
