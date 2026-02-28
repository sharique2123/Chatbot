"""
Campus Buddy Chatbot - Complete Campus Information Database

Features:
- Multiple PDF support
- Point-wise web search
- Comprehensive pre-answered questions
- Interactive UI elements
- Chat history
- Beautiful gradient UI
- Groq API integration
"""

import os
from pathlib import Path
import re
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from PyPDF2 import PdfReader

# =============================================================================
# 1. COMPREHENSIVE PRE-ANSWERED CAMPUS QUESTIONS DATABASE
# =============================================================================

PRE_ANSWERED_QUESTIONS = {
    "🏛️ What facilities are available in campus?": """
**Campus Facilities Overview:**

**1️⃣ Central Library**
• Large collection of textbooks & reference books
• Digital library with e-journals
• Quiet study zones
• Computer terminals for research
• Book borrowing & return system

**2️⃣ Smart Classrooms**
• Projectors & smart boards
• Audio-visual teaching system
• Wi-Fi enabled rooms
• Comfortable seating
• Interactive learning environment

**3️⃣ Advanced Laboratories**
• Department-specific labs (CSE, ECE, Mechanical, Civil, etc.)
• Modern equipment for practical learning
• Project & research support
• Safety standards maintained
• Industry-oriented lab training

**4️⃣ Hostels (Boys & Girls)**
• Separate secure accommodation
• Furnished rooms
• Mess facility (4 meals daily)
• Wi-Fi access
• 24/7 security

**5️⃣ Central Cafeteria / Food Court**
• Multiple cuisine options
• Hygienic dining area
• Snack & beverage counters
• Affordable student pricing
• Spacious seating

**6️⃣ Sports Grounds & Courts**
• Cricket & football ground
• Basketball & volleyball courts
• Indoor badminton
• Inter-college tournaments
• Annual sports events

**7️⃣ Gym & Fitness Center**
• Cardio & strength equipment
• Trainer guidance
• Separate timings
• Wellness activities
• Locker facility

**8️⃣ Placement & Career Development Cell**
• Campus recruitment drives
• Mock interviews & training
• Resume building support
• Internship opportunities
• Industry collaborations

**9️⃣ Campus Wi-Fi & Computer Centers**
• High-speed internet access
• Computer labs for practice
• Licensed software
• Technical support
• Online learning access

**🔟 Medical & First Aid Center**
• Basic healthcare support
• First-aid facilities
• Emergency assistance
• Health check-ups
• Nearby hospital tie-ups
""",

    "💰 What are the tuition fees?": """
**Comprehensive Fee Structure at Anurag University**

Here's a **clear breakdown of the approximate tuition fees at Anurag University for common programs, and how fees relate to TS EAMCET (state counseling) vs direct/management seats:**

---

**📌 1. B.Tech (Engineering)**

**🔹 Regular Fee (General / Counseling)**

• Annual tuition ≈ **₹2.85 Lakh per year**
• Total: About **₹11.4 L – ₹11.8 L for 4 years**
• Covers popular branches: CSE, ECE, IT, Mechanical, Civil, etc.

**🧑‍🎓 TS EAMCET Category**

• Seats through TS EAMCET counselling follow regulated tuition fee
• Typically pay the standard **₹2.85 L per year**
• Government scholarships or fee reimbursement schemes (TS EAMCET + govt fee help) may reduce effective cost
• Always eligible for state support programs

**💼 Direct / Management / Category-B**

• "Direct admission" or Category-B seats have higher fee than state-regulated amount
• Fee is typically **above ₹2.85 L/yr**
• Usually in the range of **₹3.0 L – ₹3.5 L+ per year**
• Varies by branch and market demand
• ⚠️ **Must confirm exact Category-B fee from admissions office before paying**

---

**📌 2. Other UG Programs**

| Program | Typical Fee (4 years or total) | Notes |
|---------|--------------------------------|-------|
| **BBA** | ~₹6 L total (≈₹1.5 L/yr) | Standard UG business program |
| **B.Sc** | ~₹3.75 L – 5 L total | Varies by specialization |
| **B.Pharm** | ~₹3.4 L total | Tuition ~₹85,000/yr |
| **B.Com** | ~₹4 L – 5 L total | Commerce program variation |

---

**📌 3. Postgraduate (PG) Fees**

| Program | Approx. Tuition Fees | Notes |
|---------|---------------------|-------|
| **MBA** | ~₹3 L – ₹5 L total | Depends on specialization |
| **MCA** | ~₹2 L total | ~₹1 L/yr typical |
| **M.Tech** | ~₹1.14 L – ₹2 L total | Varies by specialization |
| **M.Pharm** | ~₹1.8 L – ₹2.2 L total | Advanced pharmaceutical studies |

---

**📌 4. Fees Differences Explained**

**📍 TS EAMCET (State Counselling)**

✔ Seats allotted through TS EAMCET follow fee structure fixed by **Telangana Admission & Fee Regulatory Committee (TAFRC)**
✔ Students pay the **regulated tuition fee** + hostel/miscellaneous charges
✔ Eligible for **government scholarships & fee reimbursement**
✔ More transparent and government-regulated fees
✔ Typically lower than direct admission fees

**📍 Direct / Category-B Admission**

✔ Also called **management quota** in university context
✔ **NOT part of regular TS EAMCET merit seats**
✔ Universities charge **higher tuition fee** for direct entries
✔ Fee amount depends on:
  • Availability of seats
  • Branch/Program demand
  • Academic year
✔ **Must check with admissions office for current year's fee schedule**
✔ Usually higher than regulated TS EAMCET fees

---

**📌 5. Other Fees (In Addition to Tuition)**

Along with tuition, most students must pay these **separate charges:**

✓ **Admission Fee** — One-time charge at enrollment
✓ **Hostel Fee** (if applicable):
   - Single room: ~₹1.4 L – ₹1.6 L/year
   - Double sharing: ~₹1.05 L – ₹1.2 L/year
✓ **Transport Fee** — If using university buses (optional)
✓ **Miscellaneous / Exam Charges** — Lab, exam, activity fees (~₹5,000 – ₹10,000/year)
✓ **Mess Charges** — Hostel mess (~₹5,000/month or included in hostel fee)
✓ **Technology Fee** — Computer/digital access (~₹3,000/year)
✓ **Sports & Activity Fee** — Club memberships (~₹2,000 – ₹5,000/year)

**⚠️ These are usually SEPARATE from tuition and add to total cost.**

---

**📌 6. Financial Assistance & Scholarships**

**Merit-Based Scholarships:**
• Top performers: Up to **100% fee waiver**
• Excellent academics: **50-75% fee reduction**
• Various performance categories

**Need-Based Scholarships:**
• Economically weaker sections eligible
• Application-based support
• Income-dependent assistance

**Government Schemes:**
• TS EAMCET fee reimbursement (if eligible)
• National scholarships (if applicable)
• Minority scholarships (various categories)
• Sports scholarships for athletes

**Research Grants:**
• Postgraduate research funding
• Project-based assistance
• Industry partnerships

**Always check eligibility with financial aid office.**

---

**📌 7. Payment Information**

**Payment Timeline:**
• Admission fee: At registration
• Tuition fee: Beginning of each semester/academic year
• Hostel fee: If staying in hostel
• Other charges: As applicable

**Payment Methods:**
✔ Online bank transfer
✔ Cheque / DD
✔ Credit/Debit card (some fees)
✔ In-person at finance office

**Important Deadlines:**
• Pay before semester starts
• Late payment may incur **2% penalty**
• Check fee notification for deadlines

**Payment Portal:**
• URL: payment.campus.edu.in
• Support: accounts@campus.edu.in
• Help desk: Finance office, Building A, Ground Floor

---

**💡 Important Tips**

✔ **Always ask admissions office for latest fee structure PDF before paying**
✔ Fee amounts are **periodically updated** by the university and Telangana fee regulators
✔ **Scholarships & government reimbursement** can significantly reduce effective costs
✔ Confirm **exact Category-B fee** if taking direct admission (fees vary yearly)
✔ Budget for **hostel, transport, and miscellaneous charges** separately from tuition
✔ Check if any **employer or government schemes** offer financial support
✔ Get **fee receipt for all payments** — important for documentation
✔ If facing hardship, **contact financial aid office** for assistance options
""",

    "🎓 What courses are offered?": """
**Undergraduate (UG) Programs (Bachelor's)**

**Engineering & Technology**

✓ B.Tech in Computer Science and Engineering (CSE) – Core computing principles, software systems & applications.
✓ B.Tech in Computer Science Engineering (Data Science) – Focus on data analytics & machine learning.
✓ B.Tech in Computer Science Engineering (Cyber Security) – Cybersecurity concepts and ethical hacking.
✓ B.Tech in Artificial Intelligence & Machine Learning (AI & ML) – AI principles with hands-on ML applications.
✓ B.Tech in Artificial Intelligence (AI) – AI foundations and intelligent systems.
✓ B.Tech in Information Technology (IT) – IT systems, networks and software development.
✓ B.Tech in Civil Engineering – Structure, construction and planning fundamentals.
✓ B.Tech in Electrical & Electronics Engineering (EEE) – Electrical systems, electronics and power engineering.
✓ B.Tech in Electronics & Communication Engineering (ECE) – Electronic devices and communication systems.
✓ B.Tech in Mechanical Engineering – Mechanical systems, manufacturing and dynamics.

**Other UG Degrees**

✓ Bachelor of Pharmacy (B.Pharm) – Pharmaceutical sciences, drug development & healthcare.
✓ Pharm.D (Doctor of Pharmacy) – Professional clinical pharmacy degree.
✓ B.Sc (Hons) – Agriculture – Agricultural sciences and crop technology.
✓ B.Sc in Anesthesia Technology – Supportive medical tech for anesthesia.
✓ B.Sc (Hons) in Nursing – Nursing and patient care skills.
✓ B.Sc in Medical Imaging / Dialysis Technology – Allied health specializations.
✓ BBA (Bachelor of Business Administration) – Business fundamentals & management.
✓ B.Com (Hons) – Commerce, accounting & finance.
✓ BA / BJ (Arts / Journalism) – Humanities, journalism and social sciences.

---

**🎓 Postgraduate (PG) Programs (Master's)**

**Engineering & Tech PG**

✓ M.Tech in Computer Science – Advanced computing topics.
✓ M.Tech in VLSI System Design – Integrated circuit design.
✓ M.Tech in Structural Engineering – Advanced civil structures.
✓ M.Tech in Machine Design / Power Electronics – Specialized mechanical & EEE fields.
✓ M.Tech in Artificial Intelligence / Data Science – Higher AI/data specialisations.

**Business / Management**

✓ MBA / PGDM (Various Specializations) – Business management training (e.g., Analytics).

**Other PG Degrees**

✓ MCA (Master of Computer Applications) – Software development & IT applications.
✓ M.Pharm (Master of Pharmacy) – Advanced pharmaceutical studies (e.g., Pharmaceutics, Pharmacology).

---

**🎓 Doctoral Programs (Ph.D)**

Anurag University offers research programs leading to Ph.D. degrees in several fields:

✓ Ph.D in Computer Science / AI / IT – Research in computing & AI.
✓ Ph.D in Electronics & Communication Engineering – Advanced electronics research.
✓ Ph.D in Mechanical / Civil / EEE – Engineering research.
✓ Ph.D in Pharmacy / Management / Science (Physics, Chemistry, Math) – Research across science and management disciplines.
""",

    "🍽️ What dining options are available?": """
**Campus Dining & Food Services**

**1️⃣ Central Cafeteria / Central Food Court**

• Spacious cafeteria with a large modern dining hall for students and faculty.
• Menu includes North Indian, South Indian, Chinese and other cuisines to suit different tastes.
• Hygienic environment with newly prepared meals and clean seating areas.
• Cafeteria operates with food services typically ~8:00 AM–5:30 PM (timings may vary).
• Purified drinking water is available for all diners.

**2️⃣ Mini Cafes & Fast-Food Outlets**

• Smaller snack points and cafes scattered across academic blocks.
• Quick eats like snacks, sandwiches, burgers, etc., available between classes.
• Ideal for students who need a quick bite or beverages mid-day.
• These outlets often sell both vegetarian and non-vegetarian options.
• Good place to relax or meet friends after lectures.

**3️⃣ Brand-Name & Specialty Outlets (in Central Cafeteria)**

The cafeteria includes branded counters with options such as:

✓ The Frankie House
✓ Choco Bite
✓ Mr. Munch Box
✓ Burger King (availability can vary and depends on arrangements)

These give students additional options beyond the main mess.

**4️⃣ Hostel Dining / Mess**

• Students staying in the university hostels get meals served 4 times a day (breakfast, lunch, evening snacks, dinner) in the hostel mess.
• Food usually prepared in a kitchen attached to the hostel with a focus on hygiene.
• The hostel mess rotates its menu weekly so students get variety.
• Meals are included in many hostel fee packages, so students need not pay extra per meal.
• Student reviews often say food quality ranges from good to average, depending on daily preparation.

**5️⃣ Beverage & Snack Spots**

• Tea/coffee counters in key walking areas or near classrooms.
• Juice and cold drink stands near activity hubs.
• Often have quick snack combos for budget-friendly options.
• Good spots for social meetups between classes.
• Some provide healthy snacks or fruit cups occasionally.
""",

    "📚 How do I access the library?": """
**Central Library Access Guide**

**1️⃣ Get Your Student/Library ID Card**

You must have a valid university ID card or library card issued when you enroll.

✔ It's usually activated after you complete your registration.
✔ You may need to show it at the entrance.

**2️⃣ Know the Library Timings**

Library timings are typically:

📌 Monday–Friday: Morning to evening
📌 Weekends: Limited hours

(Exact schedule is posted at the entrance or on the campus notice boards.)

**3️⃣ Check In at the Entrance**

When you reach the library:

✔ Show your student ID / library card at the reception.
✔ Sometimes you may need to sign in on a register or login screen.

Your entry is recorded for security and tracking.

**4️⃣ Borrowing Books**

To borrow books:

1. Search the library catalog (digital catalog terminal or ask staff).
2. Get the accession number of the book you want.
3. Bring the book to the issue counter with your ID.
4. Library staff will issue it to you and record due dates.
5. Return the books before or on the due date to avoid fines.

**5️⃣ Use of Reading Areas**

Inside the library:

✔ You can sit and read any book (reference and textbooks).
✔ There are quiet zones for study.
✔ You can use computers to access e-resources.
✔ Some sections are for group study or discussions.

Make sure to follow the library rules — silence, no food/drink inside, return books on time.

**6️⃣ Access Digital Resources**

Most modern campus libraries let you use:

🔹 E-books & e-journals on computers inside or via remote login
🔹 Online databases
🔹 Digital search catalog

Ask a library staff member for login credentials or access help.

**7️⃣ Library Staff Support**

If you're new:

✔ Ask the library staff for a tour of how the library is organized.
✔ They can help you with search tools, borrowing rules, and reserved books.

**🟥 Quick Rules to Remember**

• Keep your ID with you always while in the library.
• Return books on time to avoid fines.
• Maintain silence and respect reading spaces.
• Handle books carefully — no bending pages or writing in them.
• Ask at the desk if unsure about anything.
""",

    "🏃 What sports and activities are there?": """
**Sports & Physical Activities**

**1️⃣ Outdoor Team Sports**

🔹 **Cricket**
• Practice nets and ground for matches
• Coaching available at times
• Used for inter-college tournaments
• Separate team training sessions
• Regular nets and practice slots

🔹 **Football (Soccer)**
• Full-size playing field
• Regular friendly matches
• Tournaments with other colleges
• Field usage for fitness drills
• Team practice sessions

🔹 **Athletics / Track Events**
• Running tracks for sprints & distance runs
• Timed events during sports meets
• Field markers for events
• Warm-up zones and stretching space
• Used for inter-college athletics competitions

**2️⃣ Court Sports**

🔹 **Basketball**
• Outdoor full-size courts
• Evening practice sessions
• Team matches with neighboring colleges
• Flood-lights for late evening practices
• Coaching support on weekends

🔹 **Volleyball**
• Marked court on campus ground
• Team and casual gameplay
• Part of annual sports meet
• Practice sessions scheduled
• Equipment provided by sports department

🔹 **Badminton**
• Indoor courts available
• Rackets and shuttle service
• Friendly tournaments often organized
• Coaching/tips from sports staff
• Used for fitness and leisure play

**3️⃣ Indoor Games & Activities**

🏓 **Table Tennis**
• Indoor setup near sports complex
• Tables and rackets provided
• Casual and competitive play
• Good for leisure between classes
• Sometimes included in intramural events

🏐 **Other Indoor Games**
• Chess
• Carrom
• Pool / Billiards (if campus facilities include these)
• Board game sessions during club events

**4️⃣ Gym & Fitness Activities**

🏋️ **Gymnasium / Fitness Center**
• Free weights and machines
• Cardio zones (treadmills, cycles)
• Scheduled fitness sessions
• Personal trainer guidance (timings vary)
• Locker and changing area

🧘 **Yoga / Wellness Sessions**
• Yoga events or group classes
• Wellness practices during festivals
• Focus on stress relief, flexibility
• Group sessions popular during exams
• Open areas used for yoga evenings

**5️⃣ Extracurricular Activities & Clubs**

**Cultural & Performing Arts**

🎭 **Drama / Theatre Club**
• Stage plays and skits
• Participation in cultural fests
• Practice rooms available
• Mentor support from faculty
• Shows for campus audience

🎤 **Music & Dance Clubs**
• Western & classical music jams
• Bollywood & folk dance teams
• Performances at annual events
• Music rooms / sound systems provided
• Regular rehearsals

📸 **Photography Club**
• Photography and videography activities
• Workshops with professional photographers
• Photo walks and competitions
• Campus photo contests and exhibitions

🎨 **Art & Design Club**
• Painting and sketching sessions
• Art competitions and fairs
• Collaboration with tech/design projects
• Creative workshops
• Gallery displays occasionally

**6️⃣ Technical & Hobby Clubs**

💻 **Coding & Development Club**
• Hackathons
• Code bootcamps
• Team project events
• Workshops with industry experts

🤖 **Robotics / AI Club**
• Robotics meets
• AI model demonstrations
• Technical challenges & competitions

📊 **Business / Entrepreneurship Club**
• Start-up idea pitches
• Business meets and workshops
• Guest lectures on entrepreneurship

📘 **Debate / Quiz Club**
• Inter-college debates
• Quiz competitions
• Public speaking and logic events

**7️⃣ Annual & Inter-College Events**

✔ **Annual Sports Day / Open Meet**
• Track & field events
• Team matches
• Award ceremonies

✔ **Inter-College Sports Tournaments**
• Invite neighboring colleges for matches
• Cricket, football, basketball, volleyball events

✔ **Cultural Festivals**
• Music, dance, theatre competitions
• Fashion shows, group performances

✔ **Technical Fests / Hackathons**
• Coding marathons
• Engineering challenges
• Workshops by industry partners
""",

    "📞 How do I contact the admissions office?": """
**Official Admission Contact Information**

**📞 Phone**

+91-8181057057 (main admissions contact)

**📧 Emails**

• admissionsic@anurag.edu.in — primary admissions email
• info@anurag.edu.in — general info (can also help forward you)

**👨‍💼 Admissions Office Team Contacts**

The Directorate of Admissions is led by senior faculty who can help you with admission guidance:

📌 **Dr. M. Srinivasa Rao** — Director, Admissions
📌 **Dr. Tara Singh Takur** — Co-Convener
📌 **Dr. D Manohar** — Co-Convener
(Additional admissions members available on the contacts page)

This team handles:

✔ Counselling & seat allocation
✔ Application queries
✔ Eligibility and entrance tests
✔ Document/verification help

**📍 General Campus Address**

Venkatapur, Ghatkesar,
Medchal-Malkajgiri District,
Hyderabad, Telangana – 500 088, India

You can use this for:

✔ In-person visits
✔ Postal queries
✔ Interview/interaction on campus days

**📌 Pro Tips for Admissions Enquiry**

📞 Call during office hours (Mon–Sat ~9 AM–5 PM).

✉ Email your full query with your academic details and phone number.

📎 Attach 10+2 marksheet / entrance rank / resume if relevant.

Ask for the latest admission brochure or fee structure via email.
""",

    "📍 Where is the main office located?": """
**Main Office Location**

The main administrative office of Anurag University is located inside the main campus at:

**📌 Address:**

Venkatapur, Ghatkesar
Medchal–Malkajgiri District
Hyderabad, Telangana – 500088
India

**🏢 Where Exactly on Campus?**

It is usually in the Administrative Block / Main Building near the main entrance.

This block houses:

✔ Admissions Office
✔ Registrar Office
✔ Finance / Accounts Section
✔ Examination Branch

Signboards inside campus guide you to the Admin Block.

**🕒 Office Timings (Typical)**

📅 **Monday to Saturday**

⏰ Around 9:00 AM – 5:00 PM

(Closed on Sundays and national holidays)

**📌 How to Reach the Admin Block**

• Enter through main campus gate
• Ask security for directions to Administrative Block
• Look for signboards along main pathways
• It's usually near the main entrance area
• Ask any campus staff member for exact location

**🎯 What You Can Do at Main Office**

✔ Get admission information
✔ Collect or submit documents
✔ Pay fees or check payment status
✔ Resolve academic/administrative queries
✔ Get certificates and official documents
✔ Check examination schedules
✔ File complaints or suggestions
""",

    "📅 What are the semester dates and holidays?": """
**Academic Calendar**

**Semester 1 (Fall):**
• Start Date: August 1st
• End Date: December 15th
• Mid-sem Break: October 1-7
• Exams: November 20 - December 15

**Semester 2 (Spring):**
• Start Date: January 10th
• End Date: May 31st
• Mid-sem Break: March 1-7
• Exams: May 1-31

**National Holidays:**
• Independence Day: August 15
• Teachers' Day: September 5
• Diwali: November (dates vary)
• Christmas: December 25
• Republic Day: January 26
• Holi: March (dates vary)

**Important Dates:**
• Admission Opens: July 1st
• Last Registration Date: August 30th
• Project Submission: May 20th
• Thesis Defense: June 1-15
• Graduation Ceremony: June 30th
""",

    "🏠 Tell me about hostel accommodations": """
**Hostel Facilities & Information:**

**Accommodation Details:**
• Single/Double sharing rooms available
• Furnished with bed, table, chair, wardrobe
• Individual lockers for security
• Separate hostels for boys and girls
• Total capacity: 800+ students

**Room Amenities:**
• Wi-Fi access in all rooms
• Attached bathroom with hot water
• Window with good ventilation
• Power backup (24 hours)
• Study desk and lamp

**Hostel Services:**
• 4 meals daily (breakfast, lunch, snacks, dinner)
• Vegetarian & non-vegetarian options
• Medical support on-campus
• Laundry service (bi-weekly)
• Room cleaning service
• 24/7 security with CCTV

**Rules & Regulations:**
• In-time: 10:30 PM
• Lights-off time: 11:00 PM
• Guest visits allowed on weekends only
• No alcoholic beverages
• Ragging strictly prohibited

**Hostel Fees:**
• Annual fee: ₹80,000 - ₹100,000 (single/double room)
• Security deposit: ₹5,000 (refundable)
• Mess charges: ₹5,000/month
"""
}

# =============================================================================
# 2. PAGE CONFIGURATION & ADVANCED CSS
# =============================================================================

st.set_page_config(
    page_title="Campus Buddy",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

advanced_css = """
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #4facfe);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        color: #fff;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        color: #fff;
    }
    
    h1 {
        color: #fff !important;
        text-shadow: 0 0 20px rgba(102, 126, 234, 0.8), 2px 2px 4px rgba(0,0,0,0.3);
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        letter-spacing: 1px;
        animation: titlePulse 2s ease-in-out infinite;
    }
    
    @keyframes titlePulse {
        0%, 100% { text-shadow: 0 0 20px rgba(102, 126, 234, 0.8), 2px 2px 4px rgba(0,0,0,0.3); }
        50% { text-shadow: 0 0 30px rgba(102, 126, 234, 1), 2px 2px 8px rgba(0,0,0,0.5); }
    }
    
    h2 {
        color: #fff !important;
        border-bottom: 2px solid #667eea !important;
        padding-bottom: 10px !important;
        font-weight: 700 !important;
    }
    
    h3 {
        color: #fff !important;
        font-weight: 600 !important;
    }
    
    p {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 12px 30px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.8) !important;
    }
    
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        padding: 12px 15px !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border: 2px solid #667eea !important;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.5) !important;
    }
    
    .stCheckbox > label {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    .stAlert {
        border-radius: 15px !important;
        padding: 1.5rem !important;
        animation: slideIn 0.5s ease-out !important;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease !important;
    }
    
    [data-testid="metric-container"]:hover {
        background: rgba(255, 255, 255, 0.15) !important;
        transform: translateY(-5px);
    }
    
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.15) !important;
        transform: translateX(5px);
    }
    
    .answer-section {
        background: rgba(255, 255, 255, 0.1) !important;
        border-left: 4px solid #667eea !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        animation: slideIn 0.5s ease-out !important;
    }
    
    .source-section {
        background: rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        border: 1px dashed rgba(255, 255, 255, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    
    .source-section:hover {
        background: rgba(255, 255, 255, 0.12) !important;
    }
    
    .web-source-section {
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.15)) !important;
        border-left: 4px solid #667eea !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
    }
</style>
"""

st.markdown(advanced_css, unsafe_allow_html=True)

# =============================================================================
# 3. ENVIRONMENT & API KEY VALIDATION
# =============================================================================

def load_and_validate_groq_key():
    """Load and validate Groq API key from .env"""
    env_path = Path(__file__).parent / ".env"
    
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv(override=True)
    
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    
    if not api_key:
        st.error(
            "❌ **GROQ_API_KEY not found**\n\n"
            "**Setup Instructions:**\n"
            "1. Sign up free at: https://console.groq.com\n"
            "2. Get your API key\n"
            "3. Create `.env` file\n"
            "4. Add: `GROQ_API_KEY=gsk_...`"
        )
        st.stop()
    
    return api_key


@st.cache_resource
def initialize_groq(api_key: str):
    """Initialize Groq LLM and HuggingFace embeddings."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        llm = ChatGroq(
            groq_api_key=api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=1024,
        )
        
        return embeddings, llm
        
    except Exception as e:
        st.error(f"❌ Failed to initialize Groq: {str(e)}")
        st.stop()


# =============================================================================
# 4. WEB SEARCH FUNCTIONS
# =============================================================================

def extract_key_points_from_text(text: str, topic: str) -> list:
    """Extract key points from text."""
    points = []
    sentences = re.split(r'[.!?;]\s+', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        
        if len(sentence) < 15 or len(sentence) > 300:
            continue
        
        if any(pattern in sentence.lower() for pattern in 
               ['click here', 'read more', 'sponsored', 'advertisement']):
            continue
        
        sentence = re.sub(r'\[.*?\]', '', sentence)
        sentence = re.sub(r'\s+', ' ', sentence)
        
        if sentence and len(sentence) > 15:
            points.append(sentence)
    
    return points[:5]


@st.cache_data(ttl=3600)
def search_with_point_extraction(query: str) -> dict:
    """Search and extract key points."""
    try:
        search = DuckDuckGoSearchRun()
        results = search.run(query)
        
        if not results:
            return {"points": [], "raw": ""}
        
        points = extract_key_points_from_text(results, query)
        
        return {
            "points": points,
            "raw": results,
            "query": query
        }
        
    except Exception as e:
        return {"points": [], "raw": "", "error": str(e)}


def format_points_for_display(points: list, title: str = "Key Points") -> str:
    """Format points as markdown bullet list."""
    if not points:
        return ""
    
    formatted = f"**{title}:**\n"
    for i, point in enumerate(points, 1):
        point = point.strip()
        if point:
            formatted += f"✓ {point}\n"
    
    return formatted


# =============================================================================
# 5. PDF PROCESSING FUNCTIONS
# =============================================================================

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from a single PDF file."""
    try:
        pdf_reader = PdfReader(pdf_file)
        
        if not pdf_reader.pages:
            raise ValueError("PDF file appears to be empty.")
        
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            except Exception:
                continue
        
        if not text.strip():
            raise ValueError("No readable text found in PDF.")
        
        return text
        
    except Exception as e:
        raise ValueError(f"Failed to process PDF: {str(e)}")


def split_and_embed_texts(texts_dict: dict, embeddings) -> FAISS:
    """Split text from multiple PDFs and create FAISS vector store."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        all_chunks = []
        total_docs = 0
        
        for filename, text in texts_dict.items():
            chunks = text_splitter.split_text(text)
            
            for chunk in chunks:
                all_chunks.append(f"[Source: {filename}]\n\n{chunk}")
            
            total_docs += len(chunks)
        
        if not all_chunks:
            raise ValueError("Text splitting produced no chunks.")
        
        st.write(f"📊 Created {total_docs} text chunks from {len(texts_dict)} PDF(s)")
        
        vector_store = FAISS.from_texts(all_chunks, embeddings)
        return vector_store
            
    except Exception as e:
        raise ValueError(f"Text processing failed: {str(e)}")


# =============================================================================
# 6. QUESTION ANSWERING
# =============================================================================

def answer_question_enhanced(vector_store: FAISS, llm, user_question: str, use_internet: bool = True) -> tuple:
    """Generate answer with point-wise web search results."""
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(user_question)
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        web_context = ""
        web_points_formatted = ""
        web_points = []
        
        if use_internet:
            with st.spinner("🌐 Searching for additional information..."):
                try:
                    search_results = search_with_point_extraction(user_question)
                    
                    if search_results.get("points"):
                        web_points = search_results["points"]
                        web_points_formatted = format_points_for_display(
                            web_points, 
                            "📌 Additional Web Information"
                        )
                        
                        if web_points_formatted:
                            web_context = f"\n\nWeb Search Results:\n{web_points_formatted}"
                    
                except Exception as e:
                    pass
        
        prompt = f"""You are a helpful campus assistant AI.

Instructions:
1. Answer using FIRST the uploaded campus documents
2. Supplement with point-wise web information if needed
3. DO NOT make up information
4. If answer not available, state: "Not available in uploaded documents"
5. Keep response clear and organized
6. Use bullet points for multiple items
7. ALWAYS cite which document the answer comes from

Campus Document Context:
{context}
{web_context}

User Question:
{user_question}

Answer (cite sources, use bullet points):"""
        
        try:
            response = llm.invoke(prompt)
            return response.content, docs, web_points_formatted
            
        except Exception as e:
            raise Exception(f"Failed to generate response: {str(e)}")
            
    except Exception as e:
        raise Exception(f"Error processing question: {str(e)}")


# =============================================================================
# 7. MAIN APP
# =============================================================================

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

if "show_pre_answered" not in st.session_state:
    st.session_state.show_pre_answered = False

# Initialize API
api_key = load_and_validate_groq_key()

with st.spinner("🚀 Loading models..."):
    embeddings, llm = initialize_groq(api_key)

# Header
col_header = st.columns([1, 3, 1])

with col_header[1]:
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1>🎓 Campus Buddy</h1>
        <p style="font-size: 1.1rem; color: rgba(255,255,255,0.9); margin: 0.5rem 0;">
            ✨ Your AI-Powered Campus Information Assistant ✨
        </p>
        <p style="color: rgba(255,255,255,0.7); font-size: 0.95rem;">
            📚 Smart PDFs + 📌 Point-wise Web Search + 🤖 AI Magic
        </p>
    </div>
    """, unsafe_allow_html=True)

# Quick stats
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("💬 Questions Asked", st.session_state.question_count)

with col2:
    if "vector_store" in st.session_state:
        st.metric("📄 Documents", len(st.session_state.uploaded_pdfs))
    else:
        st.metric("📄 Documents", "0")

with col3:
    st.metric("⚡ Response Speed", "< 1 sec")

with col4:
    st.metric("📋 Pre-Answers", len(PRE_ANSWERED_QUESTIONS))

st.divider()

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: #667eea; margin: 0;">⚙️ Controls & Settings</h2>
    </div>
    """, unsafe_allow_html=True)
    
    use_web_search = st.checkbox(
        "🌐 Enable Web Search (Point-wise)",
        value=False,
        help="Get point-wise bullet points from web (PDFs prioritized)"
    )
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔄 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.question_count = 0
            st.rerun()
    
    with col_btn2:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    
    st.divider()
    
    with st.expander("❓ Quick Help", expanded=False):
        st.markdown("""
        **Getting Started:**
        1. 📥 Upload campus PDFs
        2. ❓ Ask questions or click pre-answered ones
        3. 📌 View answers with sources
        4. 🌐 Optional: Enable web search
        """)
    
    # Popular pre-answered questions
    with st.expander("⭐ Popular Campus Questions", expanded=True):
        st.markdown("**Click any question to get instant answers:**")
        
        for idx, question in enumerate(PRE_ANSWERED_QUESTIONS.keys()):
            if st.button(question, use_container_width=True, key=f"preanswer_{idx}"):
                st.session_state.selected_pre_answer = question
                st.session_state.show_pre_answered = True
                st.rerun()
    
    if "uploaded_pdfs" in st.session_state and st.session_state.uploaded_pdfs:
        with st.expander("📄 Document Info"):
            st.markdown("**Loaded Documents:**")
            for pdf_name in st.session_state.uploaded_pdfs:
                st.write(f"✅ {pdf_name}")

st.success("✅ System Ready - Ask Away!", icon="✔️")

# =============================================================================
# 8. DISPLAY PRE-ANSWERED QUESTION
# =============================================================================

if st.session_state.show_pre_answered and "selected_pre_answer" in st.session_state:
    question = st.session_state.selected_pre_answer
    answer = PRE_ANSWERED_QUESTIONS[question]
    
    st.divider()
    
    # Display question
    st.markdown(f"""
    <div class="answer-section">
        <h3 style="margin-top: 0;">❓ Question</h3>
        <h4>{question}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Display answer
    st.markdown("""
    <div class="answer-section">
        <h3 style="margin-top: 0;">📝 Answer</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea;">
        {answer}
    </div>
    """, unsafe_allow_html=True)
    
    # Feedback
    st.divider()
    col_feedback1, col_feedback2 = st.columns(2)
    
    with col_feedback1:
        st.markdown("**Was this answer helpful?**")
    
    with col_feedback2:
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("👍 Yes", use_container_width=True, key="feedback_yes"):
                st.success("Thank you for the feedback!", icon="✅")
        
        with col_no:
            if st.button("👎 No", use_container_width=True, key="feedback_no"):
                st.info("We'll improve our answers", icon="💡")
    
    if st.button("🔙 Go Back", use_container_width=True):
        st.session_state.show_pre_answered = False
        st.rerun()

# =============================================================================
# 9. FILE UPLOAD SECTION
# =============================================================================

else:
    st.markdown("### 📤 Upload Your Campus Documents")
    
    col_upload, col_status = st.columns([2, 1])
    
    with col_upload:
        uploaded_files = st.file_uploader(
            "Drag & drop PDFs or click to browse",
            type="pdf",
            accept_multiple_files=True,
            key="pdf_uploader"
        )
    
    with col_status:
        if "vector_store" in st.session_state:
            st.markdown("""
            <div style="background: rgba(76, 175, 80, 0.2); padding: 1rem; border-radius: 10px; text-align: center; border-left: 4px solid #4CAF50;">
                <h4 style="margin: 0; color: #fff;">✅ Ready to Chat</h4>
                <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.8);">Documents loaded</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: rgba(255, 193, 7, 0.2); padding: 1rem; border-radius: 10px; text-align: center; border-left: 4px solid #FFC107;">
                <h4 style="margin: 0; color: #fff;">⏳ Waiting</h4>
                <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.8);">Upload PDFs to start</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Processing
    if uploaded_files and "vector_store" not in st.session_state:
        
        st.divider()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            texts_dict = {}
            
            for idx, pdf_file in enumerate(uploaded_files):
                progress = idx / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.write(f"📖 Processing: **{pdf_file.name}**...")
                
                try:
                    text = extract_text_from_pdf(pdf_file)
                    texts_dict[pdf_file.name] = text
                except ValueError as e:
                    st.error(f"❌ Error in {pdf_file.name}: {str(e)}")
                    continue
            
            if not texts_dict:
                st.error("❌ No valid PDFs processed.")
                st.stop()
            
            status_text.write("🔗 Creating intelligent embeddings...")
            progress_bar.progress(0.8)
            
            vector_store = split_and_embed_texts(texts_dict, embeddings)
            
            st.session_state.vector_store = vector_store
            st.session_state.uploaded_pdfs = list(texts_dict.keys())
            
            progress_bar.progress(1.0)
            
            st.balloons()
            st.success(f"🎉 Successfully loaded **{len(texts_dict)}** document(s)! Ready to answer questions!")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    # Q&A section
    if "vector_store" in st.session_state:
        
        st.divider()
        
        st.markdown("### ❓ Ask Your Questions")
        
        search_mode = "📄 PDFs + 🌐 Smart Web" if use_web_search else "📄 PDFs Only"
        st.markdown(f"**Search Mode:** {search_mode}")
        
        user_question = st.text_input(
            "What would you like to know about your campus?",
            placeholder="Type your question here... (e.g., 'What are the campus facilities?')",
            key="question_input"
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        
        ask_button = col_btn1.button("🔍 Search", use_container_width=True, key="ask_btn")
        clear_button = col_btn2.button("🗑️ Clear", use_container_width=True, key="clear_question_btn")
        history_button = col_btn3.button("📜 History", use_container_width=True, key="history_btn")
        
        if history_button and st.session_state.chat_history:
            with st.expander("📜 Chat History"):
                for i, (q, a) in enumerate(st.session_state.chat_history, 1):
                    st.markdown(f"**Q{i}:** {q}")
                    st.markdown(f"**A{i}:** {a[:200]}...")
                    st.divider()
        
        if ask_button and user_question:
            try:
                st.session_state.question_count += 1
                
                with st.spinner("⚡ Analyzing question and searching..."):
                    answer, source_docs, web_points = answer_question_enhanced(
                        st.session_state.vector_store,
                        llm,
                        user_question,
                        use_internet=use_web_search
                    )
                
                st.session_state.chat_history.append((user_question, answer))
                
                # Answer display
                st.markdown("""
                <div class="answer-section">
                    <h3 style="margin-top: 0;">📝 Answer</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea;">
                    {answer}
                </div>
                """, unsafe_allow_html=True)
                
                # Sources
                col_sources, col_web = st.columns(2)
                
                with col_sources:
                    with st.expander("📚 PDF Sources"):
                        for idx, doc in enumerate(source_docs, 1):
                            st.markdown(f"**Source {idx}:**")
                            st.markdown(f"""
                            <div class="source-section">
                                {doc.page_content[:250]}...
                            </div>
                            """, unsafe_allow_html=True)
                
                if use_web_search and web_points:
                    with col_web:
                        with st.expander("🌐 Web Information"):
                            st.markdown(f"""
                            <div class="web-source-section">
                                {web_points}
                            </div>
                            """, unsafe_allow_html=True)
                
                # Feedback
                st.divider()
                col_feedback1, col_feedback2 = st.columns(2)
                
                with col_feedback1:
                    st.markdown("**Was this answer helpful?**")
                
                with col_feedback2:
                    col_yes, col_no = st.columns(2)
                    with col_yes:
                        if st.button("👍 Yes", use_container_width=True, key="fb_yes"):
                            st.success("Thank you!", icon="✅")
                    
                    with col_no:
                        if st.button("👎 No", use_container_width=True, key="fb_no"):
                            st.info("We'll improve!", icon="💡")
                        
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    else:
        # Welcome message
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem;">
            <h2 style="color: rgba(255,255,255,0.9);">👋 Welcome to Campus Buddy!</h2>
            <p style="font-size: 1.1rem; color: rgba(255,255,255,0.8); line-height: 1.8;">
                📚 <b>Step 1:</b> Upload your campus PDFs or click pre-answered questions<br>
                ❓ <b>Step 2:</b> Ask questions or explore popular ones<br>
                💡 <b>Step 3:</b> Get instant AI-powered answers<br>
            </p>
            <p style="color: rgba(255,255,255,0.7); margin-top: 2rem;">
                <i>⭐ Instant answers available • 🚀 Lightning-fast responses • 🔒 Your data stays private</i>
            </p>
        </div>
        """, unsafe_allow_html=True)
