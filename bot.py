from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import numpy as np
import joblib

# Загрузка модели
model = joblib.load("model.pkl")

# Список признаков
features = [
    "Какой ваш уровень сахара в крови? (в мг/дл, например: 90–200)",
    "Какое у вас верхнее кровяное давление? (например: 80–180)",
    "Есть ли у вас лишний вес? Введите ваш ИМТ (BMI, например: 18–50)",
    "Как часто вы употребляете сладкое? (от 0 до 10)",
    "Как часто вы испытываете жажду? (оцените от 0 до 10)",
    "Бывают ли частые мочеиспускания? (от 0 до 10)",
    "Насколько часто вы чувствуете усталость? (от 0 до 10)",
    "Сколько вам лет? (от 10 до 100)",
    "Уровень физической активности? (от 0 до 10)"
]

user_data = {}

# Стартовая команда
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_data[update.effective_chat.id] = []
    await update.message.reply_text("Привет! Я помогу оценить риск диабета. Введите значения по очереди:\n\n"
                                    f"{features[0]}")

# Обработка сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text

    if chat_id not in user_data:
        user_data[chat_id] = []

    try:
        value = float(text)
        user_data[chat_id].append(value)

        # Проверяем, не завершился ли сбор данных
        if len(user_data[chat_id]) < len(features):
            next_question = features[len(user_data[chat_id])]
            await update.message.reply_text(f"Введите значение для {next_question}:")
        else:
            # Модель предсказывает результат
            X = np.array(user_data[chat_id]).reshape(1, -1)
            prediction = model.predict(X)[0]
            result = "🟢 Низкий риск" if prediction == 0 else "🔴 Высокий риск"
            await update.message.reply_text(f"Результат: {result}")
            user_data[chat_id] = []  # сброс данных
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите число.")

# Запуск бота
def main():
    app = ApplicationBuilder().token("YOUR_BOT_TOKEN").build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Бот запущен...")
    app.run_polling()

if __name__ == "__main__":
    main()
