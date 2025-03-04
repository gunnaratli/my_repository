#!/usr/bin/env python3

import animals

m = animals.Mammals()
m.printMembers()

"""
b = animals.Birds()
b.printMembers()

f = animals.Fish()
f.printMembers()
"""

harmless_birds = animals.harmless.Birds()
harmless_birds.printMembers()

dangerous_fish = animals.dangerous.Fish()
dangerous_fish.printMembers()