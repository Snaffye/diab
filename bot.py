from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import joblib
import numpy as np

# Загрузка модели
model = joblib.load("model.pkl")

# Вопросы
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

# Хранилище данных
user_states = {}

# /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_states[chat_id] = {
        "answers": [],
        "awaiting_input": True
    }
    await update.message.reply_text("Привет! Я помогу оценить риск диабета. Введите значения по очереди:\n\n" + features[0])

# Обработка сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text

    if chat_id not in user_states:
        await update.message.reply_text("Напишите /start чтобы начать опрос.")
        return

    state = user_states[chat_id]

    if not state["awaiting_input"]:
        await update.message.reply_text("Опрос завершен. Напишите /start чтобы пройти снова.")
        return

    try:
        value = float(text)
        state["answers"].append(value)

        if len(state["answers"]) < len(features):
            next_question = features[len(state["answers"])]
            await update.message.reply_text(next_question)
        else:
            X = np.array(state["answers"]).reshape(1, -1)
            prediction = model.predict(X)[0]
            result = "🟢 Низкий риск" if prediction == 0 else "🔴 Высокий риск"
            await update.message.reply_text(f"Результат: {result}")
            state["awaiting_input"] = False  # Завершаем опрос
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите число.")

# Запуск
def main():
    app = ApplicationBuilder().token("7473045709:AAF8lTAD9t-xsEPlRAtgSwddB9TIZj1oJY0").build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Бот запущен...")
    app.run_polling()

if __name__ == "__main__":
    main()
