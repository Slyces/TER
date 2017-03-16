import tensorflow as tf

# Computational Graph (aussi appelé Neural Network)
"""
Composé de <nodes> qui prennent chacun un <tensor> en entrée et retournent un
tensor en sortie.
"""
# Exemple de noeud constant
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print("Print un Node, ça ressemble à ça :")
print(" "*4 + "-", node1)
print(" "*4 + "-", node2)
print("--> ça print les objets")

print("\nUne session est nécessaire pour encapsuler et éxécuter un graphe")
sess = tf.Session()
print("sess.run([node1, node2]):", sess.run([node1, node2]))

