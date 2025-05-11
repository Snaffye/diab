from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import joblib
import numpy as np

model = joblib.load("model.pkl")

# Вопросы и диапазоны значений
questions = [
    ("Какой ваш уровень сахара в крови? (в мг/дл, например: 90–200)", 90, 200),
    ("Какое у вас верхнее кровяное давление? (например: 80–180)", 80, 180),
    ("Есть ли у вас лишний вес? Введите ваш ИМТ (например: 18–50)", 18, 50),
    ("Как часто вы употребляете сладкое? (от 0 до 10)", 0, 10),
    ("Как часто вы испытываете жажду? (от 0 до 10)", 0, 10),
    ("Бывают ли частые мочеиспускания? (от 0 до 10)", 0, 10),
    ("Насколько часто вы чувствуете усталость? (от 0 до 10)", 0, 10),
    ("Сколько вам лет? (от 10 до 100)", 10, 100),
    ("Уровень физической активности? (от 0 до 10)", 0, 10),
]

user_states = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_states[chat_id] = {"answers": [], "step": 0}
    await update.message.reply_text("Привет! Я помогу оценить риск диабета. Введите значения по очереди:\n\n" + questions[0][0])

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text.strip()

    # Если пользователь не запускал /start
    if chat_id not in user_states:
        await update.message.reply_text("Пожалуйста, введите /start чтобы начать.")
        return

    state = user_states[chat_id]
    step = state["step"]

    try:
        value = float(text)
        min_val, max_val = questions[step][1], questions[step][2]
        if not (min_val <= value <= max_val):
            await update.message.reply_text(f"Введите число от {min_val} до {max_val}.")
            return

        state["answers"].append(value)
        state["step"] += 1

        if state["step"] < len(questions):
            await update.message.reply_text(questions[state["step"]][0])
        else:
            X = np.array(state["answers"]).reshape(1, -1)
            prediction = model.predict(X)[0]
            result = "🟢 Низкий риск" if prediction == 0 else "🔴 Высокий риск"
            await update.message.reply_text(f"Результат: {result}")
            del user_states[chat_id]  # сброс
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите число.")

def main():
    app = ApplicationBuilder().token("YOUR_BOT_TOKEN").build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()
