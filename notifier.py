from plyer import notification

APP_NAME = "Trading Bot"
ICON = None  # Pode adicionar caminho para .ico se quiser


def notify_trade(side: str, pair: str, qty: float, price: float, usd: float):
    emoji = "🟢" if side == "BUY" else "🔴"
    title = f"{emoji} {side} executado — {pair}"
    message = (
        f"Quantidade: {qty:.6f}\n"
        f"Preço:      ${price:,.2f}\n"
        f"Valor:      ${usd:,.2f}"
    )
    try:
        notification.notify(
            title=title,
            message=message,
            app_name=APP_NAME,
            timeout=8,
        )
    except Exception:
        pass
