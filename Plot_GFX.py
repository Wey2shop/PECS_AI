import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation

# Load the initial CSV file
df = pd.read_csv('LOGS/training.log')

# Create a figure with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

# Set up the plot lines for training accuracy and validation accuracy
training_acc_line, = ax1.plot(df['epoch'], df['accuracy'], label='Training Accuracy')
val_acc_line, = ax1.plot(df['epoch'], df['val_accuracy'], label='Validation Accuracy')
ax1.legend()
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_title('Training and Validation Accuracy')

# Set up the plot lines for training loss and validation loss
training_loss_line, = ax2.plot(df['epoch'], df['loss'], label='Training Loss')
val_loss_line, = ax2.plot(df['epoch'], df['val_loss'], label='Validation Loss')
ax2.legend()
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Training and Validation Loss')

# Define the update function to read the log file and update the plot
def update_plot(i):
    # Load the latest CSV file
    df = pd.read_csv('LOGS/training.log')

    # Update the plot lines for training accuracy and validation accuracy
    training_acc_line.set_data(df['epoch'], df['accuracy'])
    val_acc_line.set_data(df['epoch'], df['val_accuracy'])

    # Update the plot lines for training loss and validation loss
    training_loss_line.set_data(df['epoch'], df['loss'])
    val_loss_line.set_data(df['epoch'], df['val_loss'])

    # Adjust the plot limits
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()

    # Return the plot lines that have been updated
    return training_acc_line, val_acc_line, training_loss_line, val_loss_line

# Create an animation using the update function and show the plot
ani = animation.FuncAnimation(fig, update_plot, interval=10000)
plt.show()
