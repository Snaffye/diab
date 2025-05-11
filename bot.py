from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import joblib
import numpy as np

model = joblib.load("model.pkl")

questions = [
    ("Какой ваш уровень сахара в крови? (в мг/дл, например: 90–200)", 90, 200),
    ("Какое у вас верхнее кровяное давление? (например: 80–180)", 80, 180),
    ("Есть ли у вас лишний вес? Введите ваш ИМТ (например: 18–50)", 18, 50),
    ("Как часто вы употребляете сладкое? (от 0 до 10)", 0, 10),
    ("Как часто вы испытываете жажду? (от 0 до 10)", 0, 10),
    ("Бывают ли частые мочеиспускания? (от 0 до 10)", 0, 10),
    ("Насколько часто вы чувствуете усталость? (от 0 до 10)", 0, 10),
    ("Сколько вам лет? (от 10 до 100)", 10, 100),
    ("Уровень физической активности? (от 0 до 10)", 0, 10)
]

user_states = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_states[chat_id] = {
        "answers": [],
        "current_question": 0,
        "in_progress": True
    }
    await update.message.reply_text("Привет! Я помогу оценить риск диабета. Введите значения по очереди:\n\n" +
                                    questions[0][0])

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    message = update.message.text.strip()

    # Если нет состояния — просим начать
    if chat_id not in user_states or not user_states[chat_id]["in_progress"]:
        await update.message.reply_text("Напишите /start чтобы начать опрос.")
        return

    state = user_states[chat_id]
    index = state["current_question"]

    # Проверка на число
    try:
        value = float(message)
        q_text, min_val, max_val = questions[index]
        if not (min_val <= value <= max_val):
            await update.message.reply_text(f"Введите число от {min_val} до {max_val}.")
            return
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите только число.")
        return

    # Сохраняем ответ
    state["answers"].append(value)
    state["current_question"] += 1

    # Следующий вопрос или результат
    if state["current_question"] < len(questions):
        next_question = questions[state["current_question"]][0]
        await update.message.reply_text(next_question)
    else:
        X = np.array(state["answers"]).reshape(1, -1)
        prediction = model.predict(X)[0]
        result = "🟢 Низкий риск" if prediction == 0 else "🔴 Высокий риск"
        await update.message.reply_text(f"Результат: {result}")
        user_states[chat_id]["in_progress"] = False

def main():
    app = ApplicationBuilder().token("7473045709:AAF8lTAD9t-xsEPlRAtgSwddB9TIZj1oJY0").build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Бот запущен...")
    app.run_polling()

if __name__ == "__main__":
    main()
