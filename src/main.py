from fastapi import FastAPI
from api.endpoints import hackrx

app = FastAPI(
    title="Document Query System",
    description="A system for querying documents and getting answers",
    version="1.0.0"
)

app.include_router(hackrx.router, prefix="/hackrx", tags=["hackrx"])

@app.get("/")
async def root():
    return {"message": "Document Query System is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)