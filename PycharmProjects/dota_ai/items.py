items = [
    "Observer ward", "clarity", "smoke of deceit", "faerie fire", "tango", "aghanim’s shard",
    "blood grenade", "sentry ward", "enchanted mango", "dust of appearance", "healing salve",
    "bottle", "iron branch", "mante of intelligence", "circlet", "belt of strength",
    "robe of the magi", "diadem", "staff of wizardry", "gauntlets of strength",
    "slippers of agility", "band of elvenskin", "crown", "blade of alacrity", "ogre axe",
    "quelling blade", "infused raindrops", "blight stone", "gloves of haste", "quarterstaff",
    "blitz knuckles", "javelin", "mithril hammer", "ring of protection", "orb of venom",
    "blades of attack", "chainmail", "helm of iron will", "broadsword", "claymore",
    "ring of regen", "magic stick", "wind lace", "voodoo mask", "gem of true sight",
    "shadow amulet", "blink dagger", "sage’s mask", "fluffy hat", "boots of speed",
    "cloak", "morbid mask", "ghost scepter", "ring of health", "energy booster",
    "cornucopia", "talisman of evasion", "hyperstone", "demon edge", "mystic staff",
    "sacred relic", "void stone", "vitality booster", "point booster", "platemail",
    "ultimate orb", "eaglesong", "reaver", "magic wand", "null talisman", "soul ring",
    "falcon blade", "power threads", "phase boots", "hand of midas", "boots of travel",
    "helm of the overlord", "bracer", "wraith band", "orb of corrosion", "perseverance",
    "oblivion staff", "mask of madness", "helm of the dominator", "moon shard", "buckler",
    "ring of basilius", "tranquil boots", "arcane boots", "drum of endurance", "holy locket",
    "spirit vessel", "wraith pact", "guardian greaves", "headdress", "urn of shadows",
    "medallion of courage", "pavise", "mekansm", "vladimir’s offering", "pipe of insight",
    "boots of bearing", "veil of discord", "force staff", "eul’s scepter of divinity",
    "solar crest", "dagon", "aghanim’s scepter", "octarine core", "aghanim’s blessing",
    "wind waker", "glimmer cape", "aether lens", "witch blade", "rod of atos",
    "orchid malevolence", "refresher orb", "scythe of vyse", "gleipnir", "vanguard",
    "aeon disk", "eternal shroud", "lotus orb", "bloodstone", "linken’s sphere",
    "shiva’s guard", "assault cuirass", "blade mail", "soul booster", "crimson guard",
    "black king bar", "hurricane pike", "manta style", "heart of tarrasque", "crystalys",
    "armlet of mordiggian", "shadow blade", "battle fury", "ethereal blade", "butterfly",
    "daedalus", "divine rapier", "revenant’s brooch", "bloodthorn", "meteor hammer",
    "skull basher", "desolator", "nullifier", "radiance", "monkey king bar", "silver edge",
    "disperser", "abyssal blade", "dragon lance", "sange", "phylactery", "echo sabre",
    "mage slayer", "kaya and sange", "yasha and kaya", "satanic", "mjollnir", "overwhelming blink",
    "kaya", "yasha", "diffusal blade", "maelstrom", "heaven’s halberd", "sange and yasha",
    "harpoon", "eye of skadi", "arcane blink", "swift blink"
]

# Убираем кавычки и преобразуем названия
items_processed = [item.replace("’s", "s").replace(" ", "_").lower() for item in items]

# Сортируем и создаем словарь
sorted_items = sorted(items_processed)
items_dict = {item: index+1 for index, item in enumerate(sorted_items)}

print(items_dict)