from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

# 470MB
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")


def model_pipeline(text: str, image: Image):
    # prepare inputs
    encoding = processor(image, text, return_tensors="pt")

    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()

    return  model.config.id2label[idx]

# Define the Pydantic model for user registration and login
class UserCreateSchema(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLoginSchema(BaseModel):
    email: EmailStr
    password: str

# Define Pydantic models for other operations (Todo and Contact Form)
class TodoSchema(BaseModel):
    name: str
    email: EmailStr
    message: str

class UpdateTodoSchema(BaseModel):
    name: Optional[str]
    email: Optional[EmailStr]
    message: Optional[str]

class TodoResponse(BaseModel):
    id: str
    name: str
    email: EmailStr
    message: str

class Config:
        form_mode = True

class ContactFormSchema(BaseModel):
    name: str
    email: EmailStr
    message: str
    
# Enrollment schema
class Enrollment(BaseModel):
    name: str
    phone: int = Field(..., description="Phone number must be an integer")
    email: EmailStr
    address: str


class TodoSchema(BaseModel):
    name: str
    description: str
    message: str

    class Config:
        json_schema_extra = {
            "example": {
                "name": "John Doe",
                "description": "A sample description",
                "message": "Hello, this is a sample message."
            }
        }

class UpdateTodoSchema(BaseModel):
    name: Optional[str]
    description: Optional[str]
    message: Optional[str]

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Jane Doe",
                "description": "Updated description",
                "message": "This is an updated message."
            }
        }
