from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.route import router

app = FastAPI()


# Register routes
app.include_router(router, prefix="/api", tags=["Todos"])

# Root endpoint
@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "IISC PROJECT 2025"}
