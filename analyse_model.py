from tensorflow.keras.models import load_model


model = load_model("saved_model/mlp.keras")
'''for layer in model.layers:
    weights = layer.get_weights()
    print(f"Layer {layer.name} weights: {weights}")'''
model.summary()