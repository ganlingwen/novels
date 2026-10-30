extends RefCounted
## 不依赖场景树的成长与战斗规则。世界负责选中目标，本文件负责数值。

const MAX_LEVEL = 10
const ATTRS = ["body", "strength", "agility", "spirit"]
const ATTR_NAMES = {"body": "体魄", "strength": "力量", "agility": "敏捷", "spirit": "精神"}
const SLOTS = ["weapon", "armor", "head", "boots", "charm", "tool"]
const SLOT_NAMES = {"weapon": "武器", "armor": "护甲", "head": "头部", "boots": "鞋",
	"charm": "饰品", "tool": "工具"}
const QUALITIES = ["普通", "精良", "稀有", "卓越"]
const GEAR = [
	{"id": "pipe", "name": "钢管", "slot": "weapon", "power": 5, "reach": 76},
	{"id": "axe", "name": "消防斧", "slot": "weapon", "power": 12, "reach": 82},
	{"id": "spear", "name": "加固长矛", "slot": "weapon", "power": 8, "reach": 118},
	{"id": "pistol", "name": "旧式手枪", "slot": "weapon", "power": 5, "reach": 245},
	{"id": "rifle", "name": "巡逻步枪", "slot": "weapon", "power": 10, "reach": 290},
	{"id": "coat", "name": "厚帆布外套", "slot": "armor", "hp": 25, "armor": 2},
	{"id": "vest", "name": "防刺背心", "slot": "armor", "hp": 35, "armor": 4},
	{"id": "cap", "name": "工地头盔", "slot": "head", "hp": 15, "armor": 1},
	{"id": "visor", "name": "防护面罩", "slot": "head", "hp": 20, "armor": 2},
	{"id": "shoes", "name": "运动鞋", "slot": "boots", "speed": 12},
	{"id": "boots", "name": "轻型作战靴", "slot": "boots", "speed": 18, "armor": 1},
	{"id": "coil", "name": "感应线圈", "slot": "charm", "spell": 6},
	{"id": "pendant", "name": "旧金属吊坠", "slot": "charm", "hp": 20, "spell": 3},
	{"id": "kit", "name": "急救包", "slot": "tool", "healing": 15},
	{"id": "sink", "name": "散热组件", "slot": "tool", "cooling": 2}
]
const TALENTS = [
	{"id": "vigor", "name": "坚韧", "desc": "生命上限 +30", "branch": "生存"},
	{"id": "plate", "name": "稳固", "desc": "护甲 +3", "branch": "生存", "requires": "vigor"},
	{"id": "aid", "name": "急救", "desc": "药品额外恢复 25 生命", "branch": "生存", "requires": "vigor"},
	{"id": "edge", "name": "武器训练", "desc": "普通攻击伤害 +20%", "branch": "武器"},
	{"id": "tempo", "name": "节奏", "desc": "普通攻击冷却 -15%", "branch": "武器", "requires": "edge"},
	{"id": "riposte", "name": "闪身反击", "desc": "闪避后下一次普攻 +60%", "branch": "武器", "requires": "edge"},
	{"id": "conduct", "name": "导能", "desc": "异能伤害 +20%", "branch": "雷电"},
	{"id": "chain", "name": "连锁", "desc": "电弧多跳 2 个目标；与聚能互斥", "branch": "雷电",
		"requires": "conduct", "excludes": "focus"},
	{"id": "focus", "name": "聚能", "desc": "电弧仅命中单体，伤害 ×2.4；与连锁互斥", "branch": "雷电",
		"requires": "conduct", "excludes": "chain"},
	{"id": "cool", "name": "平稳呼吸", "desc": "每秒额外消退 3 点短期负荷", "branch": "负荷"},
	{"id": "stun", "name": "余震", "desc": "震雷额外眩晕 1.2 秒", "branch": "负荷", "requires": "cool"},
	{"id": "ward", "name": "护盾回流", "desc": "护盾吸收量 +35", "branch": "负荷", "requires": "cool"}
]
const SKILLS = [
	{"name": "雷击", "key": "1", "cd": 2.8, "load": 15.0, "reach": 270.0, "mult": 1.6,
		"desc": "指向雷击 / 单体伤害"},
	{"name": "连锁电弧", "key": "2", "cd": 6.0, "load": 24.0, "reach": 240.0, "mult": 1.15,
		"desc": "最多 3 个目标 / 邻近传导"},
	{"name": "震雷", "key": "3", "cd": 8.0, "load": 28.0, "reach": 130.0, "mult": 1.1,
		"desc": "周围范围伤害 / 眩晕"},
	{"name": "蓄能护盾", "key": "4", "cd": 12.0, "load": 18.0, "reach": 0.0, "mult": 0.0,
		"desc": "吸收伤害 / 持续 5 秒"}
]

var level = 1
var xp = 0
var attribute_points = 0
var talent_points = 0
var attributes = {"body": 0, "strength": 0, "agility": 0, "spirit": 0}
var talents: Array = []
var inventory: Array = []
var equipped: Dictionary = {}
var completions = 0

func _init() -> void:
	inventory = [{"id": "pipe", "quality": 0}, {"id": "coat", "quality": 0},
		{"id": "shoes", "quality": 0}]
	for item in inventory:
		equipped[template(item.id).slot] = item.duplicate()

func template(id: String) -> Dictionary:
	for item in GEAR:
		if item.id == id:
			return item
	return {}

func talent(id: String) -> Dictionary:
	for entry in TALENTS:
		if entry.id == id:
			return entry
	return {}

func has_talent(id: String) -> bool:
	return id in talents

func xp_needed() -> int:
	return 30 + level * 20

func gain_xp(amount: int) -> int:
	var gained = 0
	if level >= MAX_LEVEL:
		return gained
	xp += amount
	while level < MAX_LEVEL and xp >= xp_needed():
		xp -= xp_needed()
		level += 1
		attribute_points += 3
		talent_points += 1
		gained += 1
	if level == MAX_LEVEL:
		xp = 0
	return gained

func allocate(id: String) -> bool:
	if attribute_points <= 0 or id not in ATTRS:
		return false
	attributes[id] += 1
	attribute_points -= 1
	return true

func talent_error(id: String) -> String:
	var entry = talent(id)
	if entry.is_empty():
		return "不存在的天赋"
	if has_talent(id):
		return "已经学习"
	if talent_points <= 0:
		return "需要天赋点"
	if entry.has("requires") and not has_talent(entry.requires):
		return "前置：" + talent(entry.requires).name
	if entry.has("excludes") and has_talent(entry.excludes):
		return "与已选天赋互斥"
	return ""

func learn(id: String) -> bool:
	if not talent_error(id).is_empty():
		return false
	talents.append(id)
	talent_points -= 1
	return true

func respec() -> void:
	attribute_points = (level - 1) * 3
	talent_points = level - 1
	for key in ATTRS:
		attributes[key] = 0
	talents.clear()

func equip(index: int) -> bool:
	if index < 0 or index >= inventory.size():
		return false
	var item: Dictionary = inventory[index]
	equipped[template(item.id).slot] = item.duplicate()
	return true

func item_name(item: Dictionary) -> String:
	return QUALITIES[int(item.quality)] + " · " + template(item.id).name

func item_stats(item: Dictionary) -> Dictionary:
	var base = template(item.id)
	var result: Dictionary = {}
	var scale = 1.0 + int(item.quality) * 0.3
	for key in ["power", "hp", "armor", "spell", "speed", "healing", "cooling"]:
		if base.has(key):
			result[key] = float(base[key]) * scale
	return result

func stats() -> Dictionary:
	var result = {
		"hp": 150.0 + attributes.body * 12.0,
		"stamina": 100.0 + attributes.body * 4.0,
		"power": 20.0 + attributes.strength * 2.8,
		"spell": 24.0 + attributes.spirit * 3.2,
		"speed": 178.0, "armor": 0.0, "healing": 65.0, "cooling": 9.0,
		"attack_cd": maxf(0.28, 0.70 / (1.0 + attributes.agility * 0.045)),
		"dash_cd": maxf(0.8, 2.5 / (1.0 + attributes.agility * 0.05)),
		"reach": 76.0
	}
	for slot in equipped:
		var item: Dictionary = equipped[slot]
		var bonuses = item_stats(item)
		for key in bonuses:
			result[key] += bonuses[key]
		if slot == "weapon":
			result.reach = float(template(item.id).reach)
	if has_talent("vigor"):
		result.hp += 30.0
	if has_talent("plate"):
		result.armor += 3.0
	if has_talent("aid"):
		result.healing += 25.0
	if has_talent("edge"):
		result.power *= 1.2
	if has_talent("tempo"):
		result.attack_cd *= 0.85
	if has_talent("conduct"):
		result.spell *= 1.2
	if has_talent("cool"):
		result.cooling += 3.0
	return result

func snapshot() -> Dictionary:
	return {"level": level, "xp": xp, "attribute_points": attribute_points,
		"talent_points": talent_points, "attributes": attributes.duplicate(true),
		"talents": talents.duplicate(), "inventory": inventory.duplicate(true),
		"equipped": equipped.duplicate(true), "completions": completions}

func restore(data: Dictionary) -> bool:
	# 在独立临时对象中校验；错误存档不得部分覆盖现有角色。
	var required = ["level", "xp", "attribute_points", "talent_points", "attributes",
		"talents", "inventory", "equipped", "completions"]
	for key in required:
		if not data.has(key):
			return false
	for key in ["level", "xp", "attribute_points", "talent_points", "completions"]:
		if not (data[key] is float or data[key] is int):
			return false
		if float(data[key]) != floor(float(data[key])) or data[key] < 0:
			return false
	if data.level < 1 or data.level > MAX_LEVEL:
		return false
	if not data.attributes is Dictionary or not data.talents is Array:
		return false
	if not data.inventory is Array or not data.equipped is Dictionary:
		return false
	var spent = 0
	for key in ATTRS:
		var value = data.attributes.get(key, -1)
		if not (value is int or value is float) or value < 0 or value != floor(value):
			return false
		spent += int(value)
	if spent + int(data.attribute_points) != (int(data.level) - 1) * 3:
		return false
	if data.talents.size() + int(data.talent_points) != int(data.level) - 1:
		return false
	var seen: Array = []
	for id in data.talents:
		if not id is String or talent(id).is_empty() or id in seen:
			return false
		var entry = talent(id)
		if entry.has("requires") and entry.requires not in data.talents:
			return false
		if entry.has("excludes") and entry.excludes in data.talents:
			return false
		seen.append(id)
	if data.inventory.size() > 120:
		return false
	for item in data.inventory:
		if not valid_item(item):
			return false
	for slot in data.equipped:
		var item = data.equipped[slot]
		if slot not in SLOTS or not valid_item(item):
			return false
		if template(item.id).slot != slot or item not in data.inventory:
			return false
	if not data.equipped.has("weapon"):
		return false
	if int(data.level) < MAX_LEVEL and int(data.xp) >= 30 + int(data.level) * 20:
		return false
	if int(data.level) == MAX_LEVEL and int(data.xp) != 0:
		return false
	level = int(data.level)
	xp = int(data.xp)
	attribute_points = int(data.attribute_points)
	talent_points = int(data.talent_points)
	for key in ATTRS:
		attributes[key] = int(data.attributes[key])
	talents = data.talents.duplicate()
	inventory = data.inventory.duplicate(true)
	equipped = data.equipped.duplicate(true)
	completions = int(data.completions)
	return true

func valid_item(item: Variant) -> bool:
	if not item is Dictionary:
		return false
	if not item.get("id") is String or template(item.id).is_empty():
		return false
	var quality = item.get("quality", -1)
	return (quality is int or quality is float) and quality == floor(quality) and quality >= 0 and quality <= 3
