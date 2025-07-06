import asyncio, sys, pathlib, json

sys.path.insert(0, "src")
from risk.event_bus import RiskEventBus, RiskEvent, EventType, EventPriority
async def demo():
    bus = RiskEventBus(max_workers=1)
    await bus.start()
    await bus.publish(
        RiskEvent(
            event_type = EventType.HEARTBEAT,
            priority   = EventPriority.LOW,
            source     = "demo",
            data       = {"msg": "hi"}
        )
    )
    await asyncio.sleep(0.05)
    await bus.stop()

asyncio.run(demo())
print("Audit:", pathlib.Path("logs/risk_audit.jsonl").read_text().splitlines()[-1])


