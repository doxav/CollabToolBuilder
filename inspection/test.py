import pickle

with open('results.pkl', 'rb') as file:
    state = pickle.load(file)

print(type(state[0]))
