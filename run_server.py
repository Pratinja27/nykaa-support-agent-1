import uvicorn

from config import API_HOST, pick_free_port


def main():
    port = pick_free_port()
    print("Nykaa Support Agent")
    print("API  http://%s:%s" % (API_HOST, port))
    print("docs http://%s:%s/docs" % (API_HOST, port))
    print("mcp  http://%s:%s/mcp" % (API_HOST, port))
    uvicorn.run("api.main:app", host=API_HOST, port=port)


if __name__ == "__main__":
    main()
