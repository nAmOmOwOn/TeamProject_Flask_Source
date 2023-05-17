from random import *


def My_lotto():
  total = [x for x in range(1, 46)]
  result = []
  shuffle(total)

  for x in range(0, 6):
    x = total.pop()
    result.append(x)

  return result
