import tensorflow as tf

# Computational Graph (aussi appelé Neural Network)
"""
Composé de <nodes> qui prennent chacun un <tensor> en entrée et retournent un
tensor en sortie.
"""
# Exemple de noeud constant
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)
