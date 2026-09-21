# Define a list of dictionaries
students = [
    {
        "name": "John",
        "age": 20,
        "grade": "A",
        "major": "Computer Science"
    },
    {
        "name": "Emily",
        "age": 22,
        "grade": "B+",
        "major": "Mathematics"
    },
    {
        "name": "Sarah",
        "age": 21,
        "grade": "A-",
        "major": "Physics"
    }
]

# Save the list of dictionaries to a text file
filename = "student_data.txt"

with open(filename, "w") as file:
    for student in students:
        for key, value in student.items():
            file.write(f"{key}: {value}\n")
        file.write("\n")

# Read the list of dictionaries from the text file and display its contents
loaded_students = []
current_student = {}

with open(filename, "r") as file:
    for line in file:
        line = line.strip()
        if line == "":
            loaded_students.append(current_student)
            current_student = {}
        else:
            key, value = line.split(": ")
            current_student[key] = value

if current_student:
    loaded_students.append(current_student)

print("Loaded Student Details:")
for student in loaded_students:
    for key, value in student.items():
        print(f"{key}: {value}")
    print()
