
import pandas as pd
from pathlib import Path

# Create our own training dataset
data = [
    # Artificial Intelligence
    ("Artificial intelligence enables machines to perform intelligent tasks.", "AI"),
    ("Machine learning allows computers to learn patterns from data.", "AI"),
    ("Neural networks are inspired by the structure of the human brain.", "AI"),
    ("Supervised learning uses labeled examples to train a model.", "AI"),
    ("An AI agent observes its environment and takes actions.", "AI"),
    ("Classification assigns data to predefined categories.", "AI"),
    ("Training data helps a machine learning model learn patterns.", "AI"),
    ("Model evaluation measures how well a learning algorithm performs.", "AI"),
    ("Computer vision helps computers understand images.", "AI"),
    ("Natural language processing works with human language.", "AI"),

    # Python Programming
    ("Python is a programming language with readable syntax.", "Python Programming"),
    ("A Python function is a reusable block of code.", "Python Programming"),
    ("A for loop repeats instructions over a sequence.", "Python Programming"),
    ("A list stores multiple values in a single variable.", "Python Programming"),
    ("A dictionary stores information as key and value pairs.", "Python Programming"),
    ("Python uses indentation to define code blocks.", "Python Programming"),
    ("A class defines the structure of objects in Python.", "Python Programming"),
    ("An exception can be handled using try and except.", "Python Programming"),
    ("A module contains reusable Python code.", "Python Programming"),
    ("The pandas library helps process tabular data.", "Python Programming"),

    # Cloud Computing
    ("Cloud computing provides computing resources over the internet.", "Cloud Computing"),
    ("Infrastructure as a Service provides virtual computing resources.", "Cloud Computing"),
    ("Platform as a Service provides an environment for application development.", "Cloud Computing"),
    ("Software as a Service delivers applications over the internet.", "Cloud Computing"),
    ("Virtual machines run separate operating systems on shared hardware.", "Cloud Computing"),
    ("Cloud storage allows files to be stored on remote servers.", "Cloud Computing"),
    ("A public cloud provides services to multiple customers.", "Cloud Computing"),
    ("A private cloud is dedicated to a particular organization.", "Cloud Computing"),
    ("Containers package applications with their dependencies.", "Cloud Computing"),
    ("Scalability allows cloud resources to increase or decrease as needed.", "Cloud Computing"),

    # Cybersecurity
    ("Encryption protects information by converting it into coded data.", "Cybersecurity"),
    ("A firewall monitors and filters network traffic.", "Cybersecurity"),
    ("Phishing attempts trick users into revealing sensitive information.", "Cybersecurity"),
    ("Multi factor authentication adds extra identity verification.", "Cybersecurity"),
    ("Malware is software designed to perform harmful activities.", "Cybersecurity"),
    ("A strong password should be difficult to guess.", "Cybersecurity"),
    ("Access control determines who can use a resource.", "Cybersecurity"),
    ("A security vulnerability can be exploited by an attacker.", "Cybersecurity"),
    ("Regular backups help recover data after data loss.", "Cybersecurity"),
    ("HTTPS protects communication between a browser and a website.", "Cybersecurity"),

    # Data Science
    ("Data science extracts useful insights from collected data.", "Data Science"),
    ("Data cleaning identifies and corrects errors in a dataset.", "Data Science"),
    ("Exploratory data analysis helps discover patterns in data.", "Data Science"),
    ("A dataset can contain rows and columns of information.", "Data Science"),
    ("A bar chart compares values across different categories.", "Data Science"),
    ("The mean is calculated by dividing the sum by the count.", "Data Science"),
    ("Missing values must be handled during data preparation.", "Data Science"),
    ("Correlation measures the relationship between two variables.", "Data Science"),
    ("Data visualization presents information using charts and graphs.", "Data Science"),
    ("A test dataset evaluates a model on unseen examples.", "Data Science"),
]

# Save the dataset without overwriting an existing file
output_file = Path("dataset/documents_dataset.csv")
output_file.parent.mkdir(parents=True, exist_ok=True)

if output_file.exists():
    print("Dataset already exists. It was not overwritten.")
else:
    df = pd.DataFrame(data, columns=["text", "category"])
    df.to_csv(output_file, index=False)
    print("Dataset created successfully!")

print("Dataset location:", output_file)
print("Number of examples:", len(pd.read_csv(output_file)))
print("\nExamples per category:")
print(pd.read_csv(output_file)["category"].value_counts())
