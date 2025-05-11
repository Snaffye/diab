from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import numpy as np
from sklearn.linear_model import LogisticRegression

# Модель для предсказания
model = LogisticRegression()

# Пример данных для обучения
X_train = [
    [90, 120, 23, 5, 4, 5, 5, 30, 4],
    [120, 130, 30, 6, 3, 6, 5, 40, 3],
    [180, 140, 28, 7, 6, 7, 4, 35, 7],
    [150, 135, 25, 4, 5, 4, 5, 50, 6]
]
y_train = [0, 1, 1, 0]  # Низкий риск (0) и высокий риск (1)

model.fit(X_train, y_train)

# Список признаков
features = [
    "Какой ваш уровень сахара в крови? (в мг/дл, например: 90–200)",
    "Какое у вас верхнее кровяное давление? (например: 80–180)",
    "Есть ли у вас лишний вес? Введите ваш ИМТ (BMI, например: 18–50)",
    "Как часто вы употребляете сладкое? (от 0 до 10)",
    "Как часто вы испытываете жажду? (от 0 до 10)",
    "Бывают ли частые мочеиспускания? (от 0 до 10)",
    "Насколько часто вы чувствуете усталость? (от 0 до 10)",
    "Сколько вам лет? (от 10 до 100)",
    "Уровень физической активности? (от 0 до 10)"
]

# Словарь для хранения данных пользователя
user_data = {}

# Функция для начала опроса
def start(update: Update, context: CallbackContext):
    chat_id = update.message.chat_id
    user_data[chat_id] = []
    update.message.reply_text("Привет! Я помогу оценить риск диабета. Введите значения по очереди:")
    ask_question(update)

# Функция для запроса следующего вопроса
def ask_question(update: Update):
    chat_id = update.message.chat_id
    user_answers = user_data.get(chat_id, [])
    
    if len(user_answers) < len(features):
        question = features[len(user_answers)]
        update.message.reply_text(f"{question}")
    else:
        # Когда все ответы собраны, делаем предсказание
        X = np.array(user_answers).reshape(1, -1)
        prediction = model.predict(X)[0]
        result = "🟢 Низкий риск" if prediction == 0 else "🔴 Высокий риск"
        update.message.reply_text(f"Результат: {result}")
        user_data[chat_id] = []  # сброс данных для следующего опроса

# Функция для обработки ответов
def handle_message(update: Update, context: CallbackContext):
    chat_id = update.message.chat_id
    user_answer = update.message.text
    
    try:
        # Преобразуем ответ в число
        answer = float(user_answer)
        
        # Добавляем ответ в список
        user_data[chat_id].append(answer)
        
        # Переходим к следующему вопросу
        ask_question(update)
        
    except ValueError:
        update.message.reply_text("Пожалуйста, введите число.")
        ask_question(update)

# Основная функция
def main():
    updater = Updater("7473045709:AAF8lTAD9t-xsEPlRAtgSwddB9TIZj1oJY0", use_context=True)
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
