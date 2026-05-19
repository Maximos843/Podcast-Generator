import uvicorn
from src.config import AppConfig


def main():
    cfg = AppConfig.from_env()
    uvicorn.run("src.service.app:app", host=cfg.host, port=cfg.port, reload=(cfg.env == "dev"))


if __name__ == "__main__":
    main()
