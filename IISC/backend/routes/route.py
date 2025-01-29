import inspect
from platform import processor
from pyexpat import model
from fastapi import FastAPI, APIRouter, HTTPException, Depends , BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from bson import ObjectId
import jwt
from datetime import datetime, timedelta
from tokenizers import Encoding
from config.database import db  # connected "database.py" for MongoDB connection
from typing import List, Optional
from schema.schemas import UserCreateSchema
from schema.schemas import UserLoginSchema
from schema.schemas import TodoSchema
from schema.schemas import UpdateTodoSchema
from schema.schemas import ContactFormSchema
from schema.schemas import Enrollment
import redis
import uuid
from typing import Union
from fastapi import FastAPI, UploadFile
import io
from PIL import Image
# You can optionally add helper functions in this file if needed
from transformers import ViltProcessor, ViltForQuestionAnswering
 #  import requests
from PIL import Image

# 470MB
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# prepare image + question
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# text = "How many cats are there?"

# prepare inputs
# encoding = processor(image, text, return_tensors="pt")

# TODO: inspect encoding
# print(encoding)
# forward pass
# outputs = model(**encoding)
# logits = outputs.logits
# idx = logits.argmax(-1).item()

# TODO: inspect outputs and test how to get the answer
#print(idx)
#print()
#print(model.config.id2label))

# print("Predicted answer:", model.config.id2label[idx])

# TODO: put above code into a function that accepts image and text as input

# App Setup
router = APIRouter()

# Connect to Redis database
redis_client = redis.StrictRedis(host="localhost", port=6379, db=0, decode_responses=True)


# MongoDB collections
todos_collection = db["todos"]
users_collection = db["users"]
contacts_collection = db["contacts"]
enrollment_collection = db["eroll"]

# Password hashing setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Helper functions for password hashing and JWT creation/validation
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Helper function to return a validated dictionary for Todo
def todo_helper(todo) -> dict:
    return {
        "id": str(todo.get("_id", "")),  # Default to an empty string if "_id" is missing
        "name": todo.get("name", ""),   # Default to an empty string if "name" is missing
        "email": todo.get("email", ""), # Default to an empty string if "email" is missing
        "message": todo.get("message", "") # Default to an empty string if "message" is missing
    }

def model_pipeline(text: str, image: Image):
    # prepare inputs
    encoding = processor(image, text, return_tensors="pt")

    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()

    return  model.config.id2label[idx]


# Register User Endpoint
@router.post("/register", response_description="Register a new user")
async def register_user(user: UserCreateSchema):
    # Check if email already exists
    if users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    # Hash password and store the user in the database
    hashed_password = hash_password(user.password)
    user_data = user.dict()
    user_data["password"] = hashed_password
    result = users_collection.insert_one(user_data)

    return {"message": "User registered successfully", "id": str(result.inserted_id)}

# Login User Endpoint
@router.post("/login", response_description="Login and get token")
async def login_user(user: UserLoginSchema):
    db_user = users_collection.find_one({"email": user.email})
    if not db_user or not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Generate JWT token
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

# Protect routes with JWT token (example)
def get_current_user(token: str = Depends(verify_token)):
    payload = token
    email = payload.get("sub")
    if email is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    return email

# Todo Routes
@router.post("/todos/", response_description="Add new todo")
async def create_todo(todo: TodoSchema):
    new_todo = todo.dict()
    result = todos_collection.insert_one(new_todo)
    created_todo = todos_collection.find_one({"_id": result.inserted_id})
    return todo_helper(created_todo)

@router.get("/todos/", response_description="List all todos")
async def get_todos():
    todos = []
    for todo in todos_collection.find():
        todos.append(todo_helper(todo))
    return todos

@router.get("/todos/{id}", response_description="Get a single todo by ID")
async def get_todo_by_id(id: str):
    todo = todos_collection.find_one({"_id": ObjectId(id)})
    if not todo:
        raise HTTPException(status_code=404, detail="Todo not found")
    return todo_helper(todo)

@router.put("/todos/{id}", response_description="Update a todo by ID")
async def update_todo(id: str, todo: UpdateTodoSchema):
    update_data = {k: v for k, v in todo.dict().items() if v is not None}
    if not update_data:
        raise HTTPException(status_code=400, detail="No data to update")
    result = todos_collection.update_one({"_id": ObjectId(id)}, {"$set": update_data})
    if result.modified_count > 0:
        updated_todo = todos_collection.find_one({"_id": ObjectId(id)})
        return todo_helper(updated_todo)
    raise HTTPException(status_code=404, detail="Todo not found")

@router.delete("/todos/{id}", response_description="Delete a todo by ID")
async def delete_todo(id: str):
    result = todos_collection.delete_one({"_id": ObjectId(id)})
    if result.deleted_count > 0:
        return {"message": "Todo deleted successfully"}
    raise HTTPException(status_code=404, detail="Todo not found")

# contact form
@router.post("/contacts/", response_description="Submit a contact form")
async def submit_contact_form(contact_form: ContactFormSchema):
    contact_data = contact_form.dict()
    result = contacts_collection.insert_one(contact_data)
    created_contact = contacts_collection.find_one({"_id": result.inserted_id})
    return {
        "id": str(created_contact["_id"]),
        "name": created_contact["name"],
        "email": created_contact["email"],
        "message": created_contact["message"]
    }
    
# submissions data
@router.get("/contacts/data", response_description="List all contact form submissions")
async def get_contact_forms():
    contacts = []
    for contact in contacts_collection.find():
        contacts.append({
            "id": str(contact["_id"]),
            "name": contact["name"],
            "email": contact["email"],
            "message": contact["message"]
        })
    return contacts

@router.post("/enroll")
def enroll_user(data: Enrollment):
    """
    Enroll a new user by saving their details in Redis.
    """
    # Generate a unique ID for the enrollment
    enrollment_id = str(uuid.uuid4())

    # Save the data to Redis
    user_data = data.dict()
    redis_client.hmset(enrollment_id, user_data)

    return {"message": "Enrollment successful", "enrollment_id": enrollment_id}

@router.get("/enroll/{enrollment_id}")
def get_enrollment(enrollment_id: str):
    """
    Get the enrollment details by ID from Redis.
    """
    user_data = redis_client.hgetall(enrollment_id)

    if not user_data:
        raise HTTPException(status_code=404, detail="Enrollment not found")

    return user_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(router, host="0.0.0.0", port=8000)

@router.post("/ask")
def ask(text: str, image: UploadFile):
    content = image.file.read()
    image = Image.open(io.BytesIO(content))
    # image = Image.open(image.file)
    
    result = model_pipeline(text, image)
    return {"answer": result}