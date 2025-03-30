from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
import asyncio

app = FastAPI()

async def event_generator():
    while True:
        await asyncio.sleep(1)
        yield {"data": "Hello, world!"}

@app.get("/stream")
async def stream():
    return EventSourceResponse(event_generator())