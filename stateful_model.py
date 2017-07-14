# Load trained model with weights into session or use an active model
model = load_model(‘model_filename.hdf5’)
output_tensor = model.output
input_tensor = model.input
input_tensor_name = input_tensor.name

reset_ops = []
for layer in model.layers:
  # old_value is the state variable, new_value is the exit node
  for old_value, new_value in layer.updates:
    zeros = np.zeros(old_value.shape, dtype='float32')
    op = tf.assign(old_value, zeros)
    reset_ops.append(op)

update_ops = []
for layer in model.layers:
  for old_value, new_value in layer.updates:
    op = tf.assign(old_value, new value)
    update_ops.append(update_op) 
