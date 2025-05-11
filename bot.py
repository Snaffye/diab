from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import joblib
import numpy as np

# Загрузка модели
model = joblib.load("model.pkl")

# Список признаков с более простыми вопросами и диапазонами
features = [
    ("Какой ваш уровень сахара в крови? (в мг/дл, например: 90–200)", (90, 200)),
    ("Какое у вас верхнее кровяное давление? (например: 80–180)", (80, 180)),
    ("Есть ли у вас лишний вес? Введите ваш ИМТ (BMI, например: 18–50)", (18, 50)),
    ("Как часто вы употребляете сладкое? (от 0 до 10)", (0, 10)),
    ("Как часто вы испытываете жажду? (от 0 до 10)", (0, 10)),
    ("Бывают ли частые мочеиспускания? (от 0 до 10)", (0, 10)),
    ("Насколько часто вы чувствуете усталость? (от 0 до 10)", (0, 10)),
    ("Сколько вам лет? (от 10 до 100)", (10, 100)),
    ("Уровень физической активности? (от 0 до 10)", (0, 10)),
]

user_data = {}

# Стартовая команда
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_data[update.effective_chat.id] = []
    await update.message.reply_text("Привет! Я помогу оценить риск диабета. Введите значения по очереди:\n\n"
                                    "Какой ваш уровень сахара в крови? (в мг/дл, например: 90–200)")

# Обработка сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text

    if chat_id not in user_data:
        user_data[chat_id] = []

    try:
        # Проверка на числовое значение
        value = float(text)
        
        # Получаем текущий вопрос и его диапазон
        question, (min_value, max_value) = features[len(user_data[chat_id])]
        
        if min_value <= value <= max_value:
            user_data[chat_id].append(value)
        else:
            await update.message.reply_text(f"Пожалуйста, введите значение от {min_value} до {max_value}.")
            return

        # Если все данные введены, делаем предсказание
        if len(user_data[chat_id]) == len(features):
            X = np.array(user_data[chat_id]).reshape(1, -1)
            prediction = model.predict(X)[0]
            result = "🟢 Низкий риск" if prediction == 0 else "🔴 Высокий риск"
            await update.message.reply_text(f"Результат: {result}")
            user_data[chat_id] = []  # сброс
        else:
            # Переходим к следующему вопросу
            next_question = features[len(user_data[chat_id])][0]
            await update.message.reply_text(f"{next_question}")
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите число.")

# Запуск бота
def main():
    app = ApplicationBuilder().token("7473045709:AAF8lTAD9t-xsEPlRAtgSwddB9TIZj1oJY0").build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Бот запущен...")
    app.run_polling()

if __name__ == "__main__":
    main()
