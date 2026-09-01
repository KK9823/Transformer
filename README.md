Transformer project in progress...

Steps
- Gather and process training data (Done)
- Work on tokenizing the characters (Done)
- Embedding + Positional Layer (Done)
- Attention Layer (Done)
- Feed forward network + transformer block (Done)
- Transformer class (Done)
- Write training scripts and start training the model (Done)
- Start Tweaking Hyperparameters for better performance (In progress)

How to use
- You will want to train a model first. Run train.py using "python train.py"
  - It will ask if you want to load a model. If it's the first time running, enter no and it will start the training based on the parameters in config.py
  - You can exit the training at any time with a keyboard interrupt (Ctrl+C or press the red square if its in pycharm because that counts as a keyboard interrupt), the model will be saved to a file with the format checkpoint_n.pt where n is the amount of steps
  - IMPORTANT NOTE: Do not change the hyperparameters while you are saving and loading checkpoints. This may cause unexpected errors.
  - If you want to load and continue training, run train.py again, say yes to "Load a model?" and enter the filename (checkpoint_n.pt) when it asks
  - When the model finishes training, it will be saved in a file called model.pt
- After you have a model.pt, you can run main.py and start prompting
  - Run main.py with "python main.py" (main cannot be ran without a model.pt)
  - You can enter a prompt which the model will attempt to continue (Ex: "Once upon a time", "I woke up and found", etc.)
  - You can then enter the amount of characters the model will generate.
- How do I change the training data?
  - If you are getting ebooks from project gutenberg, you can put the raw text file into data/raw and use CleanTxtFile.txt to clean it and put it into data/processed
  - Otherwise, you will have to clean the file yourself so that it only contains the data that should be used to train, and put the file into data/processed
  - After you have your text files in data/processed, you can run the Combine.py file to combine them all into one file, it will appear as data/combined.txt
  - Or you can do all the data processing/cleaning yourself and just put it in data/combined.txt, what matters is that all the training text is there.
  - After that, run TokenizeAndSplit.py to process the data into data/data.pt so that it can be used
  - Once you have your data/data.pt, the data side of things is all done and you can go back to training.

Credits:
- Project Gutenberg for ebook files (training data)
- Books
  - Winds of rebellion : In the dungeon of the prison ship by Ernest Haycox
  - Pride and Prejudice by Jane Austen
  - Moby Dick; Or, The Whale by Herman Melville
  - The Odyssey by Homer
  - A Room with a View by E. M. Forster