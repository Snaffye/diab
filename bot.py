from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import joblib
import numpy as np

# Загрузка модели
model = joblib.load("model.pkl")

# Средние значения и стандартные отклонения по признакам
mean = np.array([121.682, 72.255, 26.606, 118.66, 32.451, 0.472, 33.241])
std = np.array([30.436, 12.116, 9.631, 93.08, 6.875, 0.331, 11.76])

# Названия признаков (без беременности)
features = [
    "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

questions = [
    "1️⃣ Укажите уровень глюкозы в крови (примерно от 90 до 200):",
    "2️⃣ Укажите ваше артериальное давление (примерно от 80 до 180):",
    "3️⃣ Укажите толщину кожной складки (примерно от 10 до 100 мм):",
    "4️⃣ Укажите уровень инсулина (примерно от 15 до 300):",
    "5️⃣ Укажите ваш ИМТ (BMI, например от 18 до 50):",
    "6️⃣ Укажите наследственный фактор (насколько у вас в семье был диабет, от 0.0 до 2.5):",
    "7️⃣ Укажите ваш возраст (например от 10 до 100):"
]

# Временное хранилище ответов пользователей
user_data = {}

# Команда /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_data[update.effective_chat.id] = []
    await update.message.reply_text("👋 Привет! Я помогу оценить риск диабета. Пожалуйста, отвечайте на следующие вопросы:")
    await update.message.reply_text(questions[0])

# Обработка пользовательских сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text

    if chat_id not in user_data:
        user_data[chat_id] = []

    try:
        value = float(text)
        user_data[chat_id].append(value)

        if len(user_data[chat_id]) < len(features):
            await update.message.reply_text(questions[len(user_data[chat_id])])
        else:
            X = np.array(user_data[chat_id]).reshape(1, -1)
            X_scaled = (X - mean) / std  # Масштабирование вручную
            prediction = model.predict(X_scaled)[0]
            result = "🟢 Низкий риск диабета" if prediction == 0 else "🔴 Высокий риск диабета"
            await update.message.reply_text(f"✅ Результат: {result}")
            user_data[chat_id] = []
    except ValueError:
        await update.message.reply_text("❗ Пожалуйста, введите числовое значение.")

# Запуск бота
def main():
    app = ApplicationBuilder().token("7473045709:AAF8lTAD9t-xsEPlRAtgSwddB9TIZj1oJY0").build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("🤖 Бот запущен...")
    app.run_polling()

if name == "main":
    main()
