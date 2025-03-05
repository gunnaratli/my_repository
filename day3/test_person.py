#!/usr/bin/env python3

#from classroom import Student 
from classroom import Person, Student, Teacher

person = Person("Gunnar", "Eggertsson")
print(person.return_full_name()) 

student = Student("Gunnar", "Eggertsson", "Geophysics")
student.printNameSubject() 

# Test example from question 1c
from classroom import Student
me = Student('Benedikt', 'Daurer', 'physics') 
me.printNameSubject() 

from classroom import Teacher
teacher = Teacher("Filipe", "Maia", "Advanced Scientific Programming with Python")
teacher.printNameTaughtCourse()