from aptavia.tuners import GeneticTuner

tuner = GeneticTuner(5)

print([x.tunings for x in tuner.population])
tuner.step()
print([x.tunings for x in tuner.population])
