import pytest, asyncio

@pytest.fixture(scope="session")
def event_loop():
    """Use a session-wide loop so async fixtures share it."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

