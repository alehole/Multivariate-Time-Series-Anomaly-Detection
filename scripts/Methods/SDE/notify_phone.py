import requests

NTFY_TOPIC = "sde-ekf-pl1-nll"


def notify_phone(message, title="Python script"):
    response = requests.post(
        f"https://ntfy.sh/{NTFY_TOPIC}",
        data=message.encode("utf-8"),
        headers={
            "Title": title,
        },
        timeout=10,
    )

    response.raise_for_status()


if __name__ == "__main__":
    notify_phone(
        "Parameter estimation has finished.",
        title="EKF finished",
    )
