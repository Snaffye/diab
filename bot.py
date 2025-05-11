from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import joblib
import numpy as np

# Загрузка модели
model = joblib.load("model.pkl")

# Список вопросов (features) с диапазонами значений
features = [
    "Какой ваш уровень сахара в крови? (в мг/дл, например: 90–200)",
    "Какое у вас верхнее кровяное давление? (например: 80–180)",
    "Есть ли у вас лишний вес? Введите ваш ИМТ (BMI, например: 18–50)",
    "Как часто вы употребляете сладкое? (от 0 — никогда до 10 — очень часто)",
    "Как часто вы испытываете жажду? (оцените от 0 — никогда до 10 — очень часто)",
    "Бывают ли частые мочеиспускания? (от 0 — нет до 10 — очень часто)",
    "Насколько часто вы чувствуете усталость? (от 0 — нет до 10 — постоянно)",
    "Сколько вам лет? (примерно от 10 до 100)",
    "Какой у вас уровень физической активности? (от 0 — нет до 10 — высокая активность)"
]

user_data = {}

# Команда /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_data[chat_id] = []  # сброс данных при старте
    await update.message.reply_text("Привет! Я помогу оценить риск диабета. Введите значения по очереди:\n\n"
                                    f"{features[0]}")

# Обработка входящих сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text

    if chat_id not in user_data:
        user_data[chat_id] = []

    try:
        value = float(text)
        user_data[chat_id].append(value)

        if len(user_data[chat_id]) < len(features):
            next_question = features[len(user_data[chat_id])]
            await update.message.reply_text(f"{next_question}")
        else:
            X = np.array(user_data[chat_id]).reshape(1, -1)
            prediction = model.predict(X)[0]
            result = "🟢 Низкий риск" if prediction == 0 else "🔴 Высокий риск"
            await update.message.reply_text(f"Результат: {result}")
            user_data[chat_id] = []  # сброс после окончания
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
