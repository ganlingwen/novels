extends SceneTree
## 使用独立存档与真实渲染，验证入口、快捷键和成长界面。
const Main = preload("res://main.tscn")
const Store = preload("res://scripts/save_store.gd")
var scene: Node
var failures = 0
var checks = 0
var path = "/tmp/graymist-ui-test-%d.json" % OS.get_process_id()

func _initialize() -> void:
	call_deferred("run")

func check(condition: bool, message: String) -> void:
	checks += 1
	if not condition:
		failures += 1
		printerr("UI FAIL: " + message)

func press(key: int) -> void:
	var event = InputEventKey.new()
	event.physical_keycode = key
	event.keycode = key
	event.pressed = true
	Input.parse_input_event(event)
	await process_frame
	event = InputEventKey.new()
	event.physical_keycode = key
	event.keycode = key
	event.pressed = false
	Input.parse_input_event(event)
	await process_frame

func capture(name: String) -> void:
	scene.refresh_hud()
	scene.queue_redraw()
	await process_frame
	await RenderingServer.frame_post_draw
	root.get_texture().get_image().save_png("/tmp/graymist-ui-" + name + ".png")

func click_control(control: Button) -> void:
	check(control.visible and not control.mouse_filter == Control.MOUSE_FILTER_IGNORE, control.name + " 可点击")
	control.pressed.emit()
	await process_frame

func run() -> void:
	scene = Main.instantiate()
	scene.store = Store.new(path)
	root.add_child(scene)
	await process_frame
	check(not scene.modal.visible, "成长面板默认隐藏")
	await capture("camp")
	await click_control(scene.mode_button)
	check(scene.world.mode == 2, "点击切换自动模式")
	await click_control(scene.base_button)
	check(scene.world.active, "点击场景入口出发")
	await click_control(scene.growth_button)
	check(scene.modal.visible and scene.world.paused, "打开成长面板并暂停")
	var modal_position: Vector2 = scene.world.player
	await press(KEY_E)
	await press(KEY_1)
	await press(KEY_TAB)
	check(scene.world.player == modal_position and scene.world.mode == 2, "面板阻止战斗快捷键")
	await press(KEY_ESCAPE)
	check(not scene.modal.visible and not scene.world.paused, "关闭面板恢复战斗")
	await press(KEY_ESCAPE)
	check(scene.world.paused, "Esc 暂停")
	var position: Vector2 = scene.world.player
	await process_frame
	check(scene.world.player == position, "暂停时实际帧不移动")
	await press(KEY_ESCAPE)
	check(not scene.world.paused, "Esc 恢复")
	scene.set_process(false)
	for i in 600:
		scene.world.tick(1.0 / 60)
	scene.refresh_sheets()
	await capture("battle")
	scene.world.spawn_enemy(4, Vector2(820, 365))
	var boss = scene.world.enemies.back()
	boss.windup = 0.7
	boss.blast = Vector2(780, 320)
	await capture("boss")
	scene.world.return_to_base(false)
	scene.character.gain_xp(200)
	scene.refresh_sheets()
	scene.open_sheet(1)
	scene.auto_allocate()
	check(scene.character.attribute_points == 0 and scene.character.attributes.spirit > 0, "推荐加点")
	await capture("attributes")
	scene.tabs.current_tab = 2
	scene.learn("conduct")
	check(scene.character.has_talent("conduct"), "界面学习天赋")
	await capture("talents")
	scene.character.inventory.append({"id": "rifle", "quality": 2})
	scene.equip_item(scene.character.inventory.size() - 1)
	scene.tabs.current_tab = 3
	check(scene.character.equipped.weapon.id == "rifle", "界面换装")
	await capture("equipment")
	scene.save_game()
	var saved = JSON.parse_string(FileAccess.get_file_as_string(path))
	check(saved.character.equipped.weapon.id == "rifle", "界面操作保存")
	DirAccess.remove_absolute(path)
	print("UI CHECKS: %d | FAILURES: %d | screenshots: /tmp/graymist-ui-*.png" % [checks, failures])
	quit(1 if failures else 0)
