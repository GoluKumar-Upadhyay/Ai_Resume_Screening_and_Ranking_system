import io
import pickle
import re

from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.pdfpage import PDFPage
from io import StringIO

# ✅ Load model and vectorizer
clf = pickle.load(open('C:/Users/DELL/OneDrive/Desktop/clf.pkl', 'rb'))
vectorizer = pickle.load(open('C:/Users/DELL/OneDrive/Desktop/vectorizer.pkl', 'rb'))
# ✅ Category Mapping (No label_encoder.pkl needed)
Category_mapping = {
    15: "Java Developer", 23: "Testing", 8: "DevOps Engineer", 20: "Python Developer",
    24: "Web Designing", 12: "HR", 13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
    18: "Operations Manager", 6: "Data Science", 22: "Sales", 16: "Mechanical Engineer",
    1: "Arts", 7: "Database", 11: "Electrical Engineering", 14: "Health and fitness",
    19: "PMO", 4: "Business Analyst", 9: "DotNet Developer", 2: "Automation Testing",
    17: "Network Security Engineer", 21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate"
}

app = Flask(__name__)

def extract_text_from_pdf_file(file_stream):
    resource_manager = PDFResourceManager()
    string_io = StringIO()
    converter = TextConverter(resource_manager, string_io)
    interpreter = PDFPageInterpreter(resource_manager, converter)
    for page in PDFPage.get_pages(file_stream, caching=True, check_extractable=True):
        interpreter.process_page(page)
    text = string_io.getvalue()
    converter.close()
    string_io.close()
    return text

def clean_resume(text):
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_skills(text):
    keywords = [
        "Python", "Java", "JavaScript", "C++", "C#", "Go", "Rust", "Ruby", "PHP", "Swift",
        "Kotlin", "TypeScript", "Scala", "R", "MATLAB", "SQL", "NoSQL", "MongoDB", "PostgreSQL", "MySQL",
        "React", "Angular", "Vue.js", "HTML", "CSS", "SASS", "Bootstrap", "Tailwind CSS", "jQuery",
        "Node.js", "Express.js", "Django", "Flask", "Spring Boot", "Laravel", "ASP.NET",
        "REST API", "GraphQL", "JSON", "AJAX", "FastAPI", "Git", "GitHub", "GitLab", "Docker",
        "Kubernetes", "Jenkins", "CI/CD", "Ansible", "Terraform", "Vagrant", "AWS", "Azure", "Google Cloud",
        "Linux", "Unix", "Windows Server", "Shell Scripting", "Bash", "PowerShell", "Machine Learning",
        "Deep Learning", "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy", "Matplotlib", "Seaborn",
        "Jupyter", "Keras", "XGBoost", "OpenCV", "Data Analysis", "Data Science", "Big Data", "Apache Spark",
        "Hadoop", "Tableau", "Power BI", "Excel", "Google Analytics", "ETL", "Airflow", "Kafka", "dbt",
        "Snowflake", "Looker", "Artificial Intelligence", "Natural Language Processing", "Computer Vision",
        "Neural Networks", "Statistics", "Probability", "Mathematics", "Algorithms", "Data Structures",
        "Problem Solving", "Project Management", "Agile", "Scrum", "Kanban", "JIRA", "Confluence", "Slack",
        "Microsoft Teams", "Zoom", "Communication", "Leadership", "Team Management", "Time Management",
        "Critical Thinking", "Analytical Thinking", "Creative Thinking", "Attention to Detail", "Multitasking",
        "Adaptability", "Collaboration", "Presentation Skills", "Decision Making", "Self-Motivation"
    ]
    return [kw for kw in keywords if kw.lower() in text.lower()]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'resume' not in request.files:
            return "❌ Please upload a resume PDF."

        resume_file = request.files['resume']
        job_description = request.form.get('job', '')

        if resume_file.filename == '':
            return "❌ No selected file."
        if not job_description.strip():
            return "❌ Please enter a job description."

        try:
            file_stream = resume_file.stream
            raw_text = extract_text_from_pdf_file(file_stream)
            cleaned = clean_resume(raw_text)

            # ✅ TF-IDF vectorization
            resume_vec = vectorizer.transform([cleaned])
            job_vec = vectorizer.transform([clean_resume(job_description)])

            # ✅ Predict category
            pred_id = clf.predict(resume_vec)[0]
            category = Category_mapping.get(pred_id, "Unknown")

            # ✅ Extract skills and match score
            skills = extract_skills(raw_text)
            similarity = cosine_similarity(job_vec, resume_vec).flatten()[0]

            return render_template("index.html",
                                   category=category,
                                   score=similarity * 100,
                                   skill=skills)
        except Exception as e:
            return f"⚠️ Error processing the resume: {str(e)}"

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
