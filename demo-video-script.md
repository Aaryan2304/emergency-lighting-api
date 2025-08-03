# Emergency Lighting Detection API - Demo Video Script
**Duration: 2 minutes**
**Target: Walkthrough of detection process and API functionality**

## 🎬 Video Structure

### **Intro (0:00 - 0:15) - 15 seconds**
**What to show:** Title screen + Quick overview
**Script:**
"Hi! I'm demonstrating the Emergency Lighting Detection API - an AI-powered system that automatically detects and categorizes emergency lighting fixtures from electrical construction blueprints using computer vision, OCR, and Large Language Models."

### **Part 1: Detection & Preprocessing Approach (0:15 - 0:45) - 30 seconds**
**What to show:** 
- Open a sample blueprint PDF from `data/` folder
- Show the API documentation at https://emergency-lighting-api.onrender.com/docs
- Briefly explain the pipeline

**Script:**
"My approach uses a 4-stage pipeline: First, PDF conversion to images with fallback mechanisms for different hosting environments. Second, computer vision detection using OpenCV to identify shaded rectangular areas representing emergency lights. Third, OCR text extraction using Tesseract to capture fixture symbols and nearby text. Let me show you the live API..."

### **Part 2: LLM Grouping Demonstration (0:45 - 1:15) - 30 seconds**
**What to show:**
- Upload a PDF via the API interface
- Show the background processing
- Display the LLM grouping results

**Script:**
"For intelligent grouping, I use Google Gemini LLM. The system structures extracted data and sends it to the LLM to classify fixtures into categories like 'Emergency Lights', 'Exit/Emergency Combo', etc. Watch as I upload a blueprint - the system processes it in the background and the LLM groups the detected fixtures based on their characteristics and extracted text."

### **Part 3: API Functionality & Hosting (1:15 - 2:00) - 45 seconds**
**What to show:**
- Demonstrate POST /api/v1/blueprints/upload endpoint
- Show GET /api/v1/blueprints/result endpoint
- Display the JSON response
- Show the live hosting on Render.com

**Script:**
"The API is hosted live on Render.com at emergency-lighting-api.onrender.com. It provides two main endpoints: POST /api/v1/blueprints/upload for uploading PDFs with background processing, and GET /api/v1/blueprints/result to retrieve the grouped results. The API returns structured JSON with fixture counts and descriptions. The system handles async processing, stores results in SQLite database, and provides real-time status tracking. Everything is deployed with memory optimization for the free tier, using Tesseract for OCR and Google Gemini for AI-powered fixture classification."

## 🎯 **Recording Steps:**

### **Preparation:**
1. Open these tabs:
   - Live API docs: https://emergency-lighting-api.onrender.com/docs
   - Sample PDF from your `data/` folder
   - Postman with your collection loaded
   - This script for reference

### **Screen Recording Sequence:**

#### **Segment 1 (0:00-0:15): Introduction**
- Record title slide or your IDE with project open
- Speak the intro script while showing project overview

#### **Segment 2 (0:15-0:45): Detection Approach**
- Screen record opening a blueprint PDF from `data/` folder
- Show the API documentation page
- Navigate through the project structure briefly
- Explain the 4-stage pipeline while showing relevant code files

#### **Segment 3 (0:45-1:15): LLM Grouping**
- Use the API docs interface or Postman
- Upload a PDF (use one from `data/` folder)
- Show the upload response
- Wait a moment, then query the result endpoint
- Show the JSON response with LLM-grouped fixtures

#### **Segment 4 (1:15-2:00): API & Hosting**
- Demonstrate both API endpoints clearly
- Show the JSON response structure
- Highlight the Render.com hosting URL
- Mention key technical features (async processing, database storage, etc.)

## 📝 **Quick Tips:**
- Speak clearly and at moderate pace
- Keep transitions smooth between segments  
- Ensure all text/code is readable on screen
- Test API endpoints beforehand to ensure they're working
- Have backup plan if API is slow (mention "processing in background")

## 🚀 **Sample Files to Use:**
- Use any PDF from your `data/` folder for demonstration
- The API should work with any of: EL-501.png, L-102.png, P-102.png etc.
- If PDF processing fails due to hosting limitations, mention this is expected on free tier

## 📤 **Upload Requirements:**
- Upload to YouTube, Loom, or Google Drive with public link
- Ensure video is accessible without login
- Include the public link in your submission
