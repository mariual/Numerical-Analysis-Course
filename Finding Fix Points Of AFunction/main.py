# Maryam Alipour | 9612037

from sympy import symbols, Eq, solve, diff, solveset, Interval, sympify


# This function checks if our function given by user has the conditions of fixed point iteration.
def checkFixedPoint(eq, a, b):

  # checks if range is a subset of domain
  x = symbols('x')
  f_a = eq.evalf(subs = {x : a})
  f_b = eq.evalf(subs = {x : b})
  dev = diff(eq, x)
  dev_0 = solve(dev, x)
  points = [a, b]
  domain = Interval(a, b)

  for i in dev_0:
    if domain.contains(i):
      points.append(i)
      
  cp = []
  for p in points :
    cp.append(eq.evalf(subs = {x : p}))

  R = Interval(min(cp), max(cp))

    # checks if the differential of the function is less than a k<1 in [a,b]
    if domain.contains(min(cp)) and domain.contains(max(cp)):
    dev2 = diff(dev, x)
    points = [a, b]
    for p in solve(dev2, x):
      if domain.contains(p):
        points.append(p)
    
    cp = []
    for p in points:
      cp.append(dev.evalf(subs = {x : p}))
    
    min_cp = min(cp)
    max_cp = max(cp)

    if min_cp > -1 and min_cp < 1 and max_cp > -1 and max_cp < 1:
      return True
    else:
      return False 
  else:
    return False

# This function finds the sequence of fixed points.
def solver(eqStr, a, b):
  x = symbols('x')
  eq = sympify(eqStr)
  if checkFixedPoint(eq, a, b):
    x0, n = map(float, input("Enter X_0 and number of repetitions:").split(" "))
    points = [x0]
    for i in range(int(n)):
      points.append(eq.evalf(subs = {x : points[-1]}))
    return points
  else:
    print("ERROR => The input function does not have a fixed point theorem condition")


solver("1/3*x**2-1/3", -1, 1)