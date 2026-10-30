extends SceneTree

const Rules = preload("res://scripts/rules.gd")
const World = preload("res://scripts/world.gd")
const Store = preload("res://scripts/save_store.gd")
var failures = 0
var checks = 0

func check(condition: bool, description: String) -> void:
	checks += 1
	if not condition:
		failures += 1
		printerr("FAIL: " + description)

func _initialize() -> void:
	test_progression()
	test_combat()
	test_persistence()
	test_auto_mission()
	print("CHECKS: %d | FAILURES: %d" % [checks, failures])
	quit(1 if failures else 0)

func test_progression() -> void:
	var r = Rules.new()
	var initial: Dictionary = r.stats()
	check(r.gain_xp(50) == 1 and r.level == 2, "经验阈值升级")
	check(r.attribute_points == 3 and r.talent_points == 1, "升级奖励点数")
	check(r.allocate("strength") and r.stats().power > initial.power, "属性实际影响攻击")
	r.gain_xp(300)
	check(r.learn("conduct") and r.learn("chain"), "天赋前置学习")
	check(not r.learn("focus"), "互斥天赋不可同时选择")
	r.inventory.append({"id": "rifle", "quality": 2})
	r.equip(r.inventory.size() - 1)
	check(r.stats().reach == 290 and r.stats().power > initial.power, "换装改变射程和伤害")
	r.respec()
	check(r.talents.is_empty() and r.attribute_points == (r.level - 1) * 3, "洗点回收所有点数")
	r.gain_xp(100000)
	check(r.level == 10 and r.xp == 0, "满级经验不溢出")

func test_combat() -> void:
	var r = Rules.new()
	var w = World.new(r, 123)
	w.depart()
	w.mode = 0
	w.enemies.clear()
	w.spawn_enemy(0, w.player + Vector2(60, 0))
	var target: Dictionary = w.enemies[0]
	check(w.attack(target), "射程内普攻成功")
	var health: float = target.hp
	check(not w.attack(target) and target.hp == health, "普攻冷却不能绕过")
	w.load_value = 99
	check(not w.cast(0, target) and w.skill_cds[0] == 0, "负荷不足不消耗冷却")
	w.load_value = 0
	check(w.cast(0, target), "技能施放成功")
	w.strike(target, 999)
	check(target.down and r.xp == 0 and w.cores == 0, "倒地不算取核或经验")
	w.player = target.pos
	w.tick(0.02)
	check(w.cores == 1 and r.xp == 16, "靠近取核才获得经验")
	w.take_core(target)
	check(w.cores == 1, "重复取核不重复结算")
	w.attack_cd = 1.0
	w.paused = true
	var before: Vector2 = w.player
	w.tick(1.0, Vector2.RIGHT)
	check(w.player == before and w.attack_cd == 1.0, "暂停冻结位置与冷却")
	check(not w.cast(3) and not w.drink() and not w.dash(Vector2.RIGHT), "暂停禁止战斗操作")
	check(not w.visible(Vector2(150, 150), Vector2(375, 150)), "墙体阻挡视线")
	var moved: Vector2 = w.move_actor(Vector2(160, 150), Vector2(400, 0), 12)
	check(moved.x < 192, "高速位移不能穿墙")
	check(not w.route(World.BASE, Vector2(840, 120)).is_empty(), "跨地图寻路连通")
	w.paused = false
	w.hurt(10000)
	check(not w.active and w.player == World.BASE and r.xp == 16, "死亡回城保留成长")
	w.depart()
	w.mode = 2
	before = w.player
	w.tick(0.03, Vector2.UP)
	check(w.player.y < before.y and w.manual_override > 0, "移动输入立即接管自动")

func test_persistence() -> void:
	var path = "/tmp/graymist-test-%d.json" % OS.get_process_id()
	var store = Store.new(path)
	var r = Rules.new()
	r.gain_xp(150)
	r.allocate("spirit")
	r.learn("conduct")
	r.completions = 2
	r.inventory.append({"id": "rifle", "quality": 3})
	r.equip(r.inventory.size() - 1)
	check(store.save_character(r) == OK, "原子保存成功")
	var copy = Rules.new()
	check(store.load_character(copy).contains("已读取"), "读取有效存档")
	check(copy.level == r.level and copy.attributes.spirit == r.attributes.spirit and
		copy.talents == r.talents and copy.equipped.weapon.id == "rifle" and copy.completions == 2,
		"存档保留成长、天赋、装备与任务")
	var bad = r.snapshot()
	bad.attributes.spirit = -100
	var snapshot = copy.snapshot()
	check(not copy.restore(bad) and copy.snapshot() == snapshot, "损坏数据不部分覆盖角色")
	DirAccess.remove_absolute(path)
	# 用专用临时目录验证坏存档隔离；不接触玩家存档。
	var corrupt_path = "/tmp/graymist-corrupt-%d.json" % OS.get_process_id()
	var file = FileAccess.open(corrupt_path, FileAccess.WRITE)
	file.store_string("{broken")
	file.close()
	var corrupt_store = Store.new(corrupt_path)
	check(corrupt_store.load_character(Rules.new()).contains("备份"), "损坏存档保留备份")
	var directory = DirAccess.open("/tmp")
	for filename in directory.get_files():
		if filename.begins_with("graymist-corrupt-%d.json" % OS.get_process_id()):
			directory.remove(filename)

func test_auto_mission() -> void:
	for seed_value in [11, 22, 33]:
		var r = Rules.new()
		var w = World.new(r, seed_value)
		w.mode = 2
		w.depart()
		var frames = 0
		while w.active and frames < 60 * 420:
			w.tick(1.0 / 60)
			frames += 1
		print("AUTO seed=%d time=%.1fs level=%d hp=%.1f cores=%d completed=%d pos=%s enemies=%d" %
			[seed_value, frames / 60.0, r.level, w.hp, w.cores, r.completions, w.player, w.enemies.size()])
		check(r.completions == 1 and not w.active, "全自动完成清怪、Boss、取核、回城：seed %d" % seed_value)
		var completed: int = r.completions
		w.interact()
		check(w.active and w.cores == 0 and r.completions == completed, "再次出发重置行动，无重复奖励")
