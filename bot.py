from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import joblib
import numpy as np

# Загрузка модели
model = joblib.load("model.pkl")

# Список признаков
features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
user_data = {}

# Стартовая команда
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_data[update.effective_chat.id] = []
    await update.message.reply_text("Привет! Я помогу оценить риск диабета. Введите значения по очереди:\n\n"
                                    "1️⃣ Сколько беременностей было?")

# Обработка сообщений
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
            await update.message.reply_text(f"Введите значение для {next_question}:")
        else:
            X = np.array(user_data[chat_id]).reshape(1, -1)
            prediction = model.predict(X)[0]
            result = "🟢 Низкий риск" if prediction == 0 else "🔴 Высокий риск"
            await update.message.reply_text(f"Результат: {result}")
            user_data[chat_id] = []  # сброс
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите число.")

# Запуск бота
def main():
    app = ApplicationBuilder().token("7617617902:AAEh6k05vDbgnLA2VESLCHIfZ0AqiPl1jDI").build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Бот запущен...")
    app.run_polling()

if __name__ == "__main__":
    main()
