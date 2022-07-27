from tensorflow.keras.metrics import AUC, TruePositives, TrueNegatives, FalsePositives, FalseNegatives

class Metrics:
    metrics = [TruePositives(name='TP'),
            TrueNegatives(name='TN'),
            FalsePositives(name='FP'),
            FalseNegatives(name='FN'),
            AUC(curve='PR', name='AUC')]
