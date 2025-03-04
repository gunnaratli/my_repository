from .die import Die
from .utils import i_just_throw_an_exception

class GameRunner:

    def __init__(self):
        self.dice = Die.create_dice(5)
        self.reset()

    def reset(self):
        self.round = 1
        self.wins = 0
        self.loses = 0

    def answer(self):
        total = sum(die.value for die in self.dice)
        return total

    def run():
        # Probably counts wins or something.
        # Great variable name, 10/10.
        c = 0
        rond  = 1
        wins  = 0
        loses = 0
        while True:
            runner = Die.create_dice(5)
            answer = sum(die.value for die in runner)

            print("Round {}\n".format(rond))

            for die in runner:
                print(die.show())

            guess = input("Sigh. What is your guess?: ")
            guess = int(guess)

            if guess == answer:
                print("Congrats, you can add like a 5 year old...")
                wins += 1
                c += 1
            else:
                print("Sorry that's wrong")
                print("The answer is: {}".format(answer))
                print("Like seriously, how could you mess that up")
                loses += 1
                c = 0
            print("Wins: {} Loses {}".format(wins, loses))
            rond += 1

            if c == 6:
                print("You won... Congrats...")
                print("The fact it took you so long is pretty sad")
                break

            prompt = input("Would you like to play again?[y/n]: ")

            if prompt == 'y' or prompt == '':
                continue
            else:
                break
