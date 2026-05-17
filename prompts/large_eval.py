"""程序化生成 ~256 个 HumanML3D 风格英文 prompts，用于大规模散度评估。"""
import itertools
import random

random.seed(42)

# 动词 × 主语 × 修饰 — 组合后人工筛掉不合理的
SUBJECTS = ["a person", "a man", "a woman"]
ACTIONS = [
    "walks forward", "walks backward", "walks in a circle", "runs forward",
    "jogs slowly", "sprints", "jumps in place", "jumps forward", "jumps to the side",
    "kneels down", "stands up", "sits on the floor", "lies on the ground",
    "throws a left hook punch", "throws a right cross punch", "throws a kick",
    "performs a high kick", "blocks an incoming punch", "ducks under an attack",
    "waves with one hand", "waves with both arms", "claps hands",
    "raises both arms above the head", "stretches the arms out wide", "stretches forward",
    "bends down to touch the toes", "twists the torso left and right", "rolls the shoulders",
    "performs jumping jacks", "does a push-up", "does three push-ups", "does a sit-up",
    "performs a squat", "lunges forward with the left leg", "lunges forward with the right leg",
    "performs a side flip", "performs a backflip", "spins around once",
    "performs a pirouette", "dances a hip-hop routine", "dances a slow ballet",
    "performs a tai chi pose", "performs slow tai chi cloud hands",
    "kicks a soccer ball", "dribbles a basketball", "shoots a basketball",
    "swings a baseball bat", "swings a golf club", "swings a tennis racket",
    "throws a javelin", "throws a frisbee", "throws a ball overhead",
    "catches a ball", "catches a frisbee in the air",
    "marches in place", "marches forward", "tiptoes across the floor",
    "skips forward", "hops on one foot", "balances on the left leg",
    "balances on the right leg", "stands on tiptoe", "crouches down",
    "performs a yoga warrior pose", "performs a downward dog pose",
    "performs an arabesque", "performs a cartwheel",
    "drinks from a glass", "eats with a fork", "writes on a board",
    "types on a keyboard", "reads a book", "picks something up from the floor",
    "places an object on the table", "opens an imaginary door", "closes the door",
    "swims forward", "performs the breaststroke",
    "rides an imaginary bike", "paddles a boat",
    "plays the violin", "plays the piano", "plays the drums", "plays an air guitar",
    "conducts an orchestra",
    "salutes", "bows", "applauds enthusiastically",
    "raises a flag", "shoots an arrow from a bow",
    "swings a sword", "blocks with a shield",
    "performs a slow boxing combo", "shadow boxes",
    "stretches the neck side to side",
    "raises the left knee high", "raises the right knee high",
    "kicks the leg backward", "swings the right arm in a full circle",
    "swings both arms together",
    "imitates a chicken flapping its wings", "imitates a monkey jumping",
    "walks like a robot", "walks like a zombie",
    "dances the moonwalk", "dances the chacha", "dances the waltz",
    "performs a tango step",
]
MODIFIERS = [
    "", "slowly", "quickly", "with great force", "gracefully",
    "with the left hand", "with the right hand",
    "to the left", "to the right", "in a straight line",
    "for several seconds", "and then stops", "and then turns around",
    "while looking up", "while looking down",
]


def gen():
    out = set()
    for subj in SUBJECTS:
        for act in ACTIONS:
            for mod in MODIFIERS:
                s = f"{subj} {act}".strip() + (f" {mod}" if mod else "")
                out.add(s)
    return sorted(out)


if __name__ == "__main__":
    prompts = gen()
    random.shuffle(prompts)
    print(f"total: {len(prompts)}")
    # Save 512 unique
    with open("prompts/large_eval.txt", "w") as f:
        f.write("\n".join(prompts[:512]))
    print("saved -> prompts/large_eval.txt (512 prompts)")
