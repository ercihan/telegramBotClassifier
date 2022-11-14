from numpy.core.numeric import outer
import API_KEY
import telebot
import ast
import pathlib
import time
import Classifier
import os
import shutil

#bot = telebot.TeleBot(API_KEY)
bot = telebot.TeleBot(API_KEY.get())

markup = telebot.types.ReplyKeyboardMarkup(row_width=2)
itembtn1 = telebot.types.KeyboardButton('/start')
itembtn2 = telebot.types.KeyboardButton('/help')
itembtn7 = telebot.types.KeyboardButton('/state')
itembtn8 = telebot.types.KeyboardButton('/reset')
itembtn3 = telebot.types.KeyboardButton('/cnn')
itembtn4 = telebot.types.KeyboardButton('/svm')
itembtn5 = telebot.types.KeyboardButton('/dt')
itembtn6 = telebot.types.KeyboardButton('/lr')
markup.add(itembtn1, itembtn2,itembtn7,itembtn8, itembtn3,itembtn4,itembtn5,itembtn6)


@bot.message_handler(commands=['sticker'])
def receive_sticker(message):
  print(message)


@bot.message_handler(commands=['start','help'])
def start(message):
  #args = message.text
  #args = args.replace("/Start","")  
  reply_message = bot.reply_to(message, "Upload a csv file with the data to be trained. Once uploaded, select a training method.", parse_mode= 'Markdown', reply_markup=markup)
  reply_message.from_user.username = message.from_user.username
  state(reply_message)

@bot.message_handler(commands=['state'])
def state(message):
  user = message.from_user.username
  train_data_exists, test_data_exists, lr_trained = checkFiles(user)

  output  = 'Current state of uploaded and trained data:' +appendNewLine()+appendNewLine()

  output += 'Training CSV file ' + appendTrueFalseEmoji(train_data_exists) + appendNewLine()
  output += 'Test CSV file ' + appendTrueFalseEmoji(test_data_exists) + appendNewLine()
  output += 'DT trained' + appendTrueFalseEmoji(lr_trained)
     
  bot.reply_to(message, output, parse_mode= 'Markdown', reply_markup=markup)


@bot.message_handler(commands=['reset'])
def reset(message):
  user = message.from_user.username
  #Delete csv files
  if os.path.isfile('./TelegramStorage/'+user+'_train.csv'):
    os.remove('./TelegramStorage/'+user+'_train.csv')
  if os.path.isfile('./TelegramStorage/'+user+'_test.csv'):
    os.remove('./TelegramStorage/'+user+'_test.csv')
  #Delete CNN folder
  if os.path.exists('./TelegramStorage/' + user):
    shutil.rmtree('./TelegramStorage/' + user)
  #Delete LR training joblib file
  if os.path.isfile('./TelegramStorage/'+user+'_dt.joblib'):
    os.remove('./TelegramStorage/'+user+'_dt.joblib')

  bot.reply_to(message, "Files reseted. Upload first a training csv file follow by the test file.", parse_mode= 'Markdown', reply_markup=markup)


@bot.message_handler(content_types=['document'])
def read_file(message):
    print("message:")
    print(message)
    print("file_id:")
    file_id = bot.get_file(message.document.file_id)
    print(file_id)
    downloaded_file = bot.download_file(file_id.file_path)

    file_extension = file_id.file_path.rsplit('.',1)[1]
    file_name = message.from_user.username
    print('File extension:'+file_extension)
    print('user:'+file_name)

    if file_extension == 'jpg':
      with open('./TelegramStorage/'+file_name+'.'+file_extension,'wb') as new_file:
            new_file.write(downloaded_file)
      if os.path.isfile('./TelegramStorage/'+file_name+'_dt.joblib') == False:
        bot.reply_to(message, "DT is not trained yet", parse_mode= 'Markdown', reply_markup=markup)
      else:
        classified,p = Classifier.classify(file_name)
        bot.reply_to(message, "This is a "+classified, parse_mode= 'Markdown', reply_markup=markup)
    
    
    else:
      if file_extension == 'csv':
        if os.path.isfile('./TelegramStorage/'+file_name+'_train.csv'):
          file_name = file_name + '_test'
        else:
          file_name = file_name + '_train'

        with open('./TelegramStorage/'+file_name+'.'+file_extension,'wb') as new_file:
            new_file.write(downloaded_file)
        bot.reply_to(message, "File received.", parse_mode= 'Markdown', reply_markup=markup)
      else:
        bot.reply_to(message, "Data must be in csv format.", parse_mode= 'Markdown', reply_markup=markup)




@bot.message_handler(commands=['cnn','svm','dt','lr'])
def train(message):

  print('starting training')
  user = message.from_user.username

  train_data_exists, test_data_exists, lr_trained = checkFiles(user)
  if train_data_exists == False or test_data_exists == False:
    reply_message = bot.reply_to(message, "Upload test data first", parse_mode= 'Markdown', reply_markup=markup)
    reply_message.from_user.username = message.from_user.username
    state(message)
  else:
    file_name = "./TelegramStorage/"+ message.from_user.username + '.csv'
    
    if message.text == '/dt':
      accuracy = Classifier.dt("./TelegramStorage/DomeAlonso_test.csv","./TelegramStorage/DomeAlonso_train.csv",user)
      plot = open('./TelegramStorage/'+user+'_dt.png', 'rb')
      bot.send_photo(message.chat.id,plot,reply_to_message_id=message.message_id, caption='Accuracy: '+ str(accuracy))
    if message.text == '/svm':
      accuracy = Classifier.svm("./TelegramStorage/DomeAlonso_test.csv","./TelegramStorage/DomeAlonso_train.csv",user)
      plot = open('./TelegramStorage/'+user+'_svm.png', 'rb')
      bot.send_photo(message.chat.id,plot,reply_to_message_id=message.message_id, caption='Accuracy: '+str(accuracy))
    if message.text == '/lr':
      accuracy = Classifier.lr("./TelegramStorage/DomeAlonso_test.csv","./TelegramStorage/DomeAlonso_train.csv",user)
      plot = open('./TelegramStorage/'+user+'_lr.png', 'rb')
      bot.send_photo(message.chat.id,plot,reply_to_message_id=message.message_id, caption='Accuracy: '+str(accuracy))
    if message.text == '/cnn':
      accuracy = Classifier.cnn("./TelegramStorage/DomeAlonso_test.csv","./TelegramStorage/DomeAlonso_train.csv",user)
      plot = open('./TelegramStorage/'+user+'_cnn.png', 'rb')
      bot.send_photo(message.chat.id,plot,reply_to_message_id=message.message_id, caption='Accuracy: '+str(accuracy))



def checkFiles(user):
  train_data_exists = os.path.isfile('./TelegramStorage/'+user+'_train.csv')
  test_data_exists = os.path.isfile('./TelegramStorage/'+user+'_test.csv')
  lr_trained = os.path.isfile('./TelegramStorage/'+user+'_dt.joblib')
  return train_data_exists, test_data_exists, lr_trained

def appendTrueFalseEmoji(condition):
  if condition:
    return b'\xE2\x9C\x85'.decode('utf-8')
  else:
    return b'\xE2\x9D\x8C'.decode('utf-8')

def appendNewLine():
  return b'\x0A'.decode('utf-8')

bot.polling()


