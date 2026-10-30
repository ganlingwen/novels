extends RefCounted
## 二维战场：手动和自动均调用 attack / cast / dash / drink。
const Rules = preload("res://scripts/rules.gd")
const SIZE = Vector2(960, 600)
const BASE = Vector2(72, 300)
const CELL = 24
const MODES = ["手动", "辅助", "自动"]
const ENEMY_TYPES = [
	{"name": "游荡尸", "hp": 48.0, "speed": 48.0, "damage": 7.0, "xp": 16, "radius": 13.0},
	{"name": "疾行尸", "hp": 36.0, "speed": 87.0, "damage": 6.0, "xp": 19, "radius": 11.0},
	{"name": "硬肩尸", "hp": 90.0, "speed": 36.0, "damage": 12.0, "xp": 25, "radius": 17.0},
	{"name": "枝印精英", "hp": 150.0, "speed": 48.0, "damage": 14.0, "xp": 45, "radius": 19.0},
	{"name": "节点守卫", "hp": 480.0, "speed": 42.0, "damage": 19.0, "xp": 120, "radius": 23.0}
]
var rules: RefCounted
var rng = RandomNumberGenerator.new()
var nav = AStarGrid2D.new()
var obstacles: Array[Rect2] = [
	Rect2(192, 96, 144, 120), Rect2(192, 408, 144, 96),
	Rect2(432, 240, 120, 120), Rect2(624, 72, 144, 120),
	Rect2(648, 432, 120, 96)
]
var player = BASE
var facing = Vector2.RIGHT
var hp = 150.0
var stamina = 100.0
var load_value = 0.0
var shield = 0.0
var shield_time = 0.0
var attack_cd = 0.0
var skill_cds: Array[float] = [0.0, 0.0, 0.0, 0.0]
var dash_cd = 0.0
var dash_time = 0.0
var dash_direction = Vector2.RIGHT
var potion_cd = 0.0
var potions = 5
var riposte = false
var mode = 1
var manual_override = 0.0
var paused = false
var active = false
var cores = 0
var boss_spawned = false
var boss_taken = false
var enemies: Array = []
var drops: Array = []
var effects: Array = []
var messages: Array[String] = []
var changed = true
var save_requested = false
var clock = 0.0
var next_id = 0
var auto_path: PackedVector2Array = []
var path_timer = 0.0
var path_target = BASE

func _init(character: RefCounted, seed_value: int = 0) -> void:
	rules = character
	if seed_value:
		rng.seed = seed_value
	else:
		rng.randomize()
	nav.region = Rect2i(0, 0, 40, 25)
	nav.cell_size = Vector2(CELL, CELL)
	nav.offset = Vector2(CELL / 2.0, CELL / 2.0)
	nav.diagonal_mode = AStarGrid2D.DIAGONAL_MODE_ONLY_IF_NO_OBSTACLES
	nav.update()
	for x in range(40):
		for y in range(25):
			var p = Vector2(x * CELL + CELL / 2.0, y * CELL + CELL / 2.0)
			if blocked(p, 23):
				nav.set_point_solid(Vector2i(x, y), true)
	restore_health()

func note(text: String) -> void:
	messages.push_front(text)
	if messages.size() > 5:
		messages.resize(5)
	changed = true

func restore_health() -> void:
	var stats = rules.stats()
	hp = stats.hp
	stamina = stats.stamina
	load_value = 0
	shield = 0
	shield_time = 0
	attack_cd = 0
	skill_cds = [0.0, 0.0, 0.0, 0.0]
	dash_cd = 0
	dash_time = 0
	potion_cd = 0
	riposte = false
	potions = 5

func depart() -> void:
	if active:
		return
	active = true
	paused = false
	player = BASE
	cores = 0
	boss_spawned = false
	boss_taken = false
	enemies.clear()
	drops.clear()
	effects.clear()
	auto_path.clear()
	path_timer = 0
	restore_health()
	var positions = [
		Vector2(210, 290), Vector2(360, 300), Vector2(396, 145), Vector2(390, 455),
		Vector2(150, 72), Vector2(105, 500), Vector2(510, 110), Vector2(575, 180),
		Vector2(590, 420), Vector2(460, 490), Vector2(820, 260), Vector2(760, 340),
		Vector2(840, 95), Vector2(865, 470), Vector2(540, 560), Vector2(325, 550),
		Vector2(595, 295), Vector2(870, 550)
	]
	for i in positions.size():
		var type = 3 if i in [10, 17] else i % 3
		spawn_enemy(type, positions[i])
	note("委托：取核至少 12 枚，消灭节点守卫，再回基地交付。")

func spawn_enemy(type: int, position: Vector2) -> void:
	var info: Dictionary = ENEMY_TYPES[type]
	next_id += 1
	enemies.append({"id": next_id, "type": type, "pos": position, "hp": info.hp,
		"max_hp": info.hp, "down": false, "taken": false, "cd": 0.7,
		"stun": 0.0, "windup": 0.0, "blast": position, "special_cd": 3.0,
		"path": PackedVector2Array(), "path_timer": rng.randf_range(0, 0.5), "flash": 0.0})

func blocked(p: Vector2, radius: float) -> bool:
	if p.x < radius or p.y < radius or p.x > SIZE.x - radius or p.y > SIZE.y - radius:
		return true
	for obstacle in obstacles:
		if obstacle.grow(radius).has_point(p):
			return true
	return false

func move_actor(position: Vector2, velocity: Vector2, radius: float) -> Vector2:
	# 分段扫过，避免闪避或低帧率时跨越墙体。
	var steps = maxi(1, ceili(velocity.length() / 6.0))
	var step = velocity / steps
	for i in steps:
		var px = position + Vector2(step.x, 0)
		if not blocked(px, radius):
			position = px
		var py = position + Vector2(0, step.y)
		if not blocked(py, radius):
			position = py
	return position

func visible(from: Vector2, to: Vector2) -> bool:
	var steps = maxi(1, ceili(from.distance_to(to) / 8.0))
	for i in range(1, steps):
		for obstacle in obstacles:
			if obstacle.has_point(from.lerp(to, float(i) / steps)):
				return false
	return true

func grid_id(p: Vector2) -> Vector2i:
	var initial = Vector2i(clampi(int(p.x / CELL), 0, 39), clampi(int(p.y / CELL), 0, 24))
	if not nav.is_point_solid(initial):
		return initial
	var best = initial
	var distance = INF
	for dx in range(-3, 4):
		for dy in range(-3, 4):
			var candidate = initial + Vector2i(dx, dy)
			if nav.is_in_boundsv(candidate) and not nav.is_point_solid(candidate):
				var d = nav.get_point_position(candidate).distance_squared_to(p)
				if d < distance:
					distance = d
					best = candidate
	return best

func route(from: Vector2, to: Vector2) -> PackedVector2Array:
	return nav.get_point_path(grid_id(from), grid_id(to))

func along_path(position: Vector2, path: PackedVector2Array) -> Vector2:
	for point in path:
		if position.distance_to(point) > 12:
			return position.direction_to(point)
	return Vector2.ZERO

func nearest_enemy(max_distance: float = INF, require_sight: bool = false) -> Dictionary:
	var best: Dictionary = {}
	var distance = max_distance
	for enemy in enemies:
		if enemy.down:
			continue
		var d = player.distance_to(enemy.pos)
		if d <= distance and (not require_sight or visible(player, enemy.pos)):
			best = enemy
			distance = d
	return best

func target_near_cursor(aim: Vector2, reach: float) -> Dictionary:
	var best: Dictionary = {}
	var distance = INF
	for enemy in enemies:
		if enemy.down or player.distance_to(enemy.pos) > reach or not visible(player, enemy.pos):
			continue
		var d = aim.distance_squared_to(enemy.pos)
		if d < distance:
			distance = d
			best = enemy
	return best

func strike(enemy: Dictionary, damage: float, electrical: bool = false) -> void:
	if enemy.is_empty() or enemy.down:
		return
	enemy.hp -= damage
	enemy.flash = 0.12
	effects.append({"kind": "number", "pos": enemy.pos, "text": str(roundi(damage)),
		"time": 0.7, "electric": electrical})
	if enemy.hp <= 0:
		enemy.hp = 0.0
		enemy.down = true
		enemy.windup = 0.0
		if enemy.type == 3:
			for other in enemies:
				if other.type < 3 and other.pos.distance_to(enemy.pos) < 220:
					other.stun = 2.5
			note("枝印节点失能，附近尸群暂时失去组织。靠近取核。")

func attack(target: Dictionary = {}) -> bool:
	if paused or not active or attack_cd > 0:
		return false
	var stats = rules.stats()
	if target.is_empty():
		target = nearest_enemy(stats.reach, true)
	if target.is_empty() or target.down or player.distance_to(target.pos) > stats.reach:
		return false
	if not visible(player, target.pos):
		return false
	attack_cd = stats.attack_cd
	facing = player.direction_to(target.pos)
	var damage: float = stats.power
	if riposte:
		damage *= 1.6
		riposte = false
	strike(target, damage)
	effects.append({"kind": "shot" if stats.reach > 150 else "slash", "from": player,
		"to": target.pos, "time": 0.16})
	return true

func cast(index: int, target: Dictionary = {}) -> bool:
	if paused or not active or index < 0 or index > 3 or skill_cds[index] > 0:
		return false
	var skill: Dictionary = Rules.SKILLS[index]
	if load_value + skill.load > 100:
		return false
	if index != 3:
		if target.is_empty():
			target = nearest_enemy(skill.reach, true)
		if target.is_empty() or target.down or player.distance_to(target.pos) > skill.reach:
			return false
		if not visible(player, target.pos):
			return false
	skill_cds[index] = skill.cd
	load_value += skill.load
	var stats = rules.stats()
	var damage: float = stats.spell * skill.mult
	if index == 3:
		shield = 50.0 + rules.attributes.spirit * 3.0 + (35.0 if rules.has_talent("ward") else 0.0)
		shield_time = 5.0
	elif index == 2:
		effects.append({"kind": "ring", "pos": player, "radius": 130.0, "time": 0.35})
		for enemy in enemies:
			if not enemy.down and player.distance_to(enemy.pos) <= 130 and visible(player, enemy.pos):
				strike(enemy, damage, true)
				enemy.stun = 0.8 + (1.2 if rules.has_talent("stun") else 0.0)
	elif index == 1:
		if rules.has_talent("focus"):
			damage *= 2.4
		var count = 1 if rules.has_talent("focus") else (5 if rules.has_talent("chain") else 3)
		var hit: Array = []
		var origin = player
		var current = target
		for i in count:
			if current.is_empty():
				break
			effects.append({"kind": "bolt", "from": origin, "to": current.pos, "time": 0.3})
			strike(current, damage, true)
			hit.append(current.id)
			origin = current.pos
			current = {}
			var distance = 160.0
			for enemy in enemies:
				var d = origin.distance_to(enemy.pos)
				if not enemy.down and enemy.id not in hit and d < distance and visible(origin, enemy.pos):
					distance = d
					current = enemy
	else:
		strike(target, damage, true)
		effects.append({"kind": "bolt", "from": player, "to": target.pos, "time": 0.25})
	return true

func dash(direction: Vector2) -> bool:
	if paused or not active or dash_cd > 0 or stamina < 30:
		return false
	stamina -= 30
	dash_time = 0.18
	dash_cd = rules.stats().dash_cd
	dash_direction = direction.normalized() if direction.length() > 0 else facing
	if rules.has_talent("riposte"):
		riposte = true
	return true

func drink() -> bool:
	if paused or not active or potions <= 0 or potion_cd > 0 or hp >= rules.stats().hp:
		return false
	potions -= 1
	potion_cd = 8.0
	hp = minf(rules.stats().hp, hp + rules.stats().healing)
	note("使用急救药品，剩余 %d 份。" % potions)
	return true

func hurt(damage: float) -> void:
	if dash_time > 0 or not active:
		return
	damage = maxf(1.0, damage - rules.stats().armor)
	var absorbed = minf(shield, damage)
	shield -= absorbed
	hp -= damage - absorbed
	effects.append({"kind": "hurt", "pos": player, "time": 0.2})
	if hp <= 0:
		return_to_base(false)
		note("行动失败：救援队带你回基地。成长保留，本次委托重置。")

func take_core(enemy: Dictionary) -> void:
	if enemy.taken or not enemy.down:
		return
	enemy.taken = true
	cores += 1
	var levels: int = rules.gain_xp(ENEMY_TYPES[enemy.type].xp)
	if levels > 0:
		hp = minf(rules.stats().hp, hp + 40)
		note("升至 Lv.%d！获得 %d 属性点、%d 天赋点。" % [rules.level, levels * 3, levels])
	if enemy.type == 4:
		boss_taken = true
		note("节点守卫已取核。可以返回基地交付委托。")
	if enemy.type >= 3 or rng.randf() < 0.36:
		var quality = 2 if enemy.type == 4 else (1 if enemy.type == 3 else 0)
		if rng.randf() < 0.2:
			quality = mini(3, quality + 1)
		# 装备表现为倒地敌人附近找到的遗物箱，不是身体生成武器。
		drops.append({"pos": enemy.pos + Vector2(8, 6),
			"item": {"id": Rules.GEAR[rng.randi_range(0, Rules.GEAR.size() - 1)].id, "quality": quality}})
	if cores >= 10 and not boss_spawned:
		boss_spawned = true
		spawn_enemy(4, Vector2(860, 365))
		note("街区深处出现节点守卫！避开蓄力冲击的圆形预警。")
	save_requested = true
	changed = true

func can_deliver() -> bool:
	return cores >= 12 and boss_taken

func return_to_base(deliver: bool) -> void:
	if deliver and not can_deliver():
		return
	if deliver:
		rules.completions += 1
		rules.gain_xp(70)
		if rules.inventory.size() < 120:
			rules.inventory.append({"id": "coil" if rules.completions == 1 else "vest",
				"quality": 2})
	active = false
	player = BASE
	enemies.clear()
	drops.clear()
	effects.clear()
	restore_health()
	save_requested = true
	changed = true
	if deliver:
		note("委托完成：上缴 %d 枚晶核，获得 70 经验与稀有装备。" % cores)

func interact() -> void:
	if paused:
		return
	if not active:
		depart()
	elif player.distance_to(BASE) < 78:
		if can_deliver():
			return_to_base(true)
		else:
			return_to_base(false)
			note("已撤回基地，保留成长；本次委托未结算。")
	else:
		note("请靠近西侧基地入口，再按 E。")

func ai_direction(delta: float) -> Vector2:
	var goal = player
	var best_distance = INF
	# 蓄力范围内优先撤离，自动也能完成 Boss 机制。
	for enemy in enemies:
		if enemy.windup > 0 and player.distance_to(enemy.blast) < 145:
			var away = Vector2.LEFT
			var safest_distance = -1.0
			for i in 16:
				var candidate = Vector2.RIGHT.rotated(TAU * i / 16.0)
				var endpoint = move_actor(player, candidate * 165, 12)
				var distance = endpoint.distance_to(enemy.blast)
				if distance > safest_distance:
					safest_distance = distance
					away = candidate
			dash(away)
			return away
	if can_deliver():
		goal = BASE
	else:
		for enemy in enemies:
			if enemy.taken:
				continue
			var d = player.distance_to(enemy.pos)
			# 近处倒地者优先取核，远处仍需接敌。
			var weight = d * (0.65 if enemy.down else 1.0)
			if weight < best_distance:
				best_distance = weight
				goal = enemy.pos
	# 满足委托后也先拾取脚边遗物，再返程。
	for drop in drops:
		if player.distance_to(drop.pos) < 110:
			goal = drop.pos
			break
	if player.distance_to(goal) < 24:
		return Vector2.ZERO
	var target = nearest_enemy(rules.stats().reach * 0.78, true)
	if not target.is_empty() and not can_deliver() and target.pos.distance_to(goal) < 20:
		return Vector2.ZERO
	path_timer -= delta
	if path_timer <= 0 or path_target.distance_to(goal) > 36 or auto_path.is_empty():
		auto_path = route(player, goal)
		# 路径首格中心可能在角色身后；从后继格开始跟随。
		if auto_path.size() > 1:
			auto_path.remove_at(0)
		path_target = goal
		path_timer = 0.45
	while not auto_path.is_empty() and player.distance_to(auto_path[0]) < 13:
		auto_path.remove_at(0)
	if auto_path.is_empty():
		return player.direction_to(goal)
	return player.direction_to(auto_path[0])

func tick(delta: float, movement: Vector2 = Vector2.ZERO) -> void:
	if paused:
		return
	clock += delta
	for effect in effects:
		effect.time -= delta
	effects = effects.filter(func(effect): return effect.time > 0)
	if not active:
		return
	var stats = rules.stats()
	attack_cd = maxf(0, attack_cd - delta)
	for i in 4:
		skill_cds[i] = maxf(0, skill_cds[i] - delta)
	dash_cd = maxf(0, dash_cd - delta)
	dash_time = maxf(0, dash_time - delta)
	potion_cd = maxf(0, potion_cd - delta)
	shield_time = maxf(0, shield_time - delta)
	if shield_time <= 0:
		shield = 0
	load_value = maxf(0, load_value - stats.cooling * delta)
	stamina = minf(stats.stamina, stamina + 24 * delta)
	manual_override = maxf(0, manual_override - delta)
	if movement.length() > 0:
		manual_override = 1.2
	var direction = movement.normalized()
	if mode == 2 and manual_override <= 0:
		direction = ai_direction(delta)
	if direction.length() > 0:
		facing = direction
	if dash_time > 0:
		direction = dash_direction * 3.3
	player = move_actor(player, direction * stats.speed * delta, 12)
	if mode > 0:
		if hp < stats.hp * 0.45:
			drink()
		if shield <= 0 and not nearest_enemy(130).is_empty():
			cast(3)
		attack()
		if load_value < 65:
			cast(2)
			cast(1)
			cast(0)
	for enemy in enemies:
		enemy.flash = maxf(0, enemy.flash - delta)
		if enemy.down:
			continue
		enemy.stun = maxf(0, enemy.stun - delta)
		enemy.cd = maxf(0, enemy.cd - delta)
		enemy.special_cd = maxf(0, enemy.special_cd - delta)
		if enemy.windup > 0:
			enemy.windup -= delta
			if enemy.windup <= 0:
				effects.append({"kind": "blast", "pos": enemy.blast, "radius": 115.0, "time": 0.35})
				if player.distance_to(enemy.blast) < 127:
					hurt(42)
					if not active:
						return
			continue
		if enemy.stun > 0:
			continue
		var distance = player.distance_to(enemy.pos)
		if enemy.type == 4 and enemy.special_cd <= 0 and distance < 180 and visible(enemy.pos, player):
			enemy.windup = 1.15
			enemy.blast = player
			enemy.special_cd = 4.0
			continue
		var info: Dictionary = ENEMY_TYPES[enemy.type]
		if distance < 270 and distance > info.radius + 15:
			enemy.path_timer -= delta
			if enemy.path_timer <= 0:
				enemy.path = route(enemy.pos, player)
				if enemy.path.size() > 1:
					enemy.path.remove_at(0)
				enemy.path_timer = 0.5
			while not enemy.path.is_empty() and enemy.pos.distance_to(enemy.path[0]) < 12:
				enemy.path.remove_at(0)
			var enemy_direction = enemy.pos.direction_to(player)
			if not enemy.path.is_empty():
				enemy_direction = enemy.pos.direction_to(enemy.path[0])
			# 尸群保留实体间距，不能叠成一个点同时造成全部接触伤害。
			var separation = Vector2.ZERO
			for other in enemies:
				if other.id == enemy.id or other.down:
					continue
				var spacing: float = info.radius + ENEMY_TYPES[other.type].radius + 5
				var gap: float = enemy.pos.distance_to(other.pos)
				if gap > 0 and gap < spacing:
					separation += other.pos.direction_to(enemy.pos) * (1.0 - gap / spacing) * 2.5
			enemy_direction = (enemy_direction + separation).limit_length()
			enemy.pos = move_actor(enemy.pos, enemy_direction * info.speed * delta, info.radius)
		if distance < info.radius + 24 and enemy.cd <= 0 and visible(player, enemy.pos):
			enemy.cd = 1.2
			hurt(info.damage)
			if not active:
				return
	# 先收集再生成 Boss，避免遍历中扩充敌人列表。
	var ready: Array = []
	for enemy in enemies:
		if enemy.down and not enemy.taken and player.distance_to(enemy.pos) < 43:
			ready.append(enemy)
	for enemy in ready:
		take_core(enemy)
	enemies = enemies.filter(func(enemy): return not enemy.taken)
	var collected: Array = []
	for drop in drops:
		if player.distance_to(drop.pos) < 40:
			if rules.inventory.size() < 120:
				rules.inventory.append(drop.item.duplicate())
				note("发现遗物：" + rules.item_name(drop.item))
				save_requested = true
			else:
				note("背包已满（120 件），请回基地整理。")
			collected.append(drop)
	for drop in collected:
		drops.erase(drop)
	if mode == 2 and manual_override <= 0 and can_deliver() and player.distance_to(BASE) < 60:
		return_to_base(true)
