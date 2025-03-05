class Person:

    """
    "Person" class which takes firstname and lastname as arguments to the constructor __init__
    """
    def __init__(self, firstname, lastname):
        self.firstname = firstname
        self.lastname = lastname

    """
    Method which returns the full name of the person as a combined string
    """
    def return_full_name(self):
        return self.firstname + " " + self.lastname


class Student(Person):

    """
    "Student" class which inherits from the "Person" class, takes the subject area as an additional argument to the constructor __init__
    """
    def __init__(self, firstname, lastname, subject_area):
        super().__init__(firstname, lastname)
        self.subject_area = subject_area

    """
    Method that prints the full name and the subject area of the student
    Note: The method return_full_name is inherited from the Person class
    """
    def printNameSubject(self):
        name_subject = self.return_full_name() + " - Subject area: " + self.subject_area
        print(name_subject)
        return name_subject


class Teacher(Person):
    """
    "Teacher" class which also inherits from "Person", takes the name of the course as an argument
    """
    def __init__(self, firstname, lastname, taught_course):
        super().__init__(firstname, lastname)
        self.taught_course = taught_course


    """
    Method that prints the full name of the teacher and the course he teaches
    """
    def printNameTaughtCourse(self):
        name_taught = self.return_full_name() + " - Course taught: " + self.taught_course
        print(name_taught)
        return name_taught