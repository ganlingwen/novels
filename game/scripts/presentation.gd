extends "res://scripts/main.gd"
## Full-window presentation. Rules, persistence and growth actions stay in main.
const ZOOM = 1.5
var ground: Texture2D
var frames: Array = []
var modal: Control
var prior_pause = false
var growth_button: Button
var objective: Label
var toast: Label
var hint: Label
var cooldowns: Array[ColorRect] = []
var cd_labels: Array[Label] = []
var last_message = ""
var toast_time = 0.0
var previous_positions = {}
var moving = {}
var player_moving = false
var anim_time = 0.0
var last_cores = 0

func _ready() -> void:
	ground = load("res://assets/ground.jpg")
	for row in 4:
		var strip: Array = []
		for col in 3:
			strip.append(load("res://assets/actor_%d_%d.png" % [row, col]))
		frames.append(strip)
	super._ready()
	refresh_hud()

func create_ui() -> void:
	# Reuse the existing theme and growth sheets, relocating them into a modal.
	super.create_ui()
	for child in ui.get_children():
		if child != tabs:
			ui.remove_child(child)
			child.queue_free()
	skill_buttons.clear()
	modal = Control.new()
	modal.size = Vector2(1440, 900)
	ui.add_child(modal)
	var dim = ColorRect.new()
	dim.color = Color(0.015, 0.025, 0.03, 0.82)
	dim.size = modal.size
	modal.add_child(dim)
	ui.remove_child(tabs)
	modal.add_child(tabs)
	tabs.position = Vector2(440, 145)
	tabs.size = Vector2(560, 620)
	tabs.set_tab_title(0, "委托")
	modal.hide()
	var close = button_at("返回街区  ×", Vector2(805, 92), Vector2(195, 42), close_modal)
	ui.remove_child(close)
	modal.add_child(close)
	health_text = label_at("", Vector2(32, 27), Vector2(300, 25), 18, C_TEXT)
	health_bar = bar_at(Vector2(32, 59), Vector2(265, 10), C_GREEN)
	stamina_bar = bar_at(Vector2(32, 76), Vector2(170, 5), C_BLUE)
	load_bar = bar_at(Vector2(210, 76), Vector2(87, 5), C_AMBER)
	health_bar.tooltip_text = "生命"
	stamina_bar.tooltip_text = "体力"
	load_bar.tooltip_text = "异能负荷"
	objective = label_at("", Vector2(1020, 32), Vector2(385, 60), 18, C_TEXT)
	objective.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	mode_button = button_at("", Vector2(1155, 110), Vector2(250, 43), cycle_mode)
	growth_button = button_at("角色 · 背包", Vector2(32, 109), Vector2(190, 43), func(): open_sheet(1))
	pause_button = button_at("Ⅱ", Vector2(1355, 819), Vector2(50, 50), toggle_pause)
	pause_button.tooltip_text = "暂停 / 继续  Esc"
	base_button = button_at("出发探索  →", Vector2(130, 482), Vector2(205, 50), base_action)
	hint = label_at("", Vector2(32, 855), Vector2(560, 25), 14, C_DIM)
	toast = label_at("", Vector2(330, 110), Vector2(780, 65), 20, C_GREEN)
	toast.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	toast.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	for i in 6:
		var index = i
		var action: Callable
		if i < 4:
			action = func(): world.cast(index)
		elif i == 4:
			action = func(): world.drink()
		else:
			action = func(): world.dash(movement_input())
		var button = button_at("", Vector2(458 + i * 82, 808), Vector2(70, 70), action)
		button.icon = make_icon(i)
		button.icon_alignment = HORIZONTAL_ALIGNMENT_CENTER
		button.tooltip_text = (Rules.SKILLS[i].name + "： " + Rules.SKILLS[i].desc) if i < 4 else ("急救药品 Q" if i == 4 else "闪避 空格")
		var key = Label.new()
		key.text = str(i + 1) if i < 4 else ("Q" if i == 4 else "␣")
		key.position = Vector2(6, 1)
		key.add_theme_font_size_override("font_size", 12)
		key.mouse_filter = Control.MOUSE_FILTER_IGNORE
		button.add_child(key)
		if i < 4:
			skill_buttons.append(button)
			var shade = ColorRect.new()
			shade.color = Color(0.01, 0.02, 0.03, 0.7)
			shade.mouse_filter = Control.MOUSE_FILTER_IGNORE
			button.add_child(shade)
			cooldowns.append(shade)
			var cd = Label.new()
			cd.position = Vector2(20, 24)
			cd.mouse_filter = Control.MOUSE_FILTER_IGNORE
			button.add_child(cd)
			cd_labels.append(cd)
	ui.move_child(modal, -1)

func make_icon(index: int) -> Texture2D:
	var shapes = [
		'<path d="M29 3 10 26h13l-4 19 20-27H26z"/>',
		'<path d="m5 13 10 7 9-13-3 17 15-4-6 18 14-9" fill="none"/>',
		'<circle cx="24" cy="24" r="18" fill="none"/><path d="m26 9-12 18h10l-2 12 12-20H24z"/>',
		'<path d="m24 4 17 7v13q-3 14-17 20Q10 38 7 24V11z" fill="none"/><path d="M24 13v20m-9-10h18" fill="none"/>',
		'<rect x="11" y="9" width="26" height="34" rx="5" fill="none"/><path d="M18 4h12M24 17v17m-8-8h16" fill="none"/>',
		'<path d="m14 8 15 16-15 16m12-32 15 16-15 16M3 24h18" fill="none"/>'
	]
	var image = Image.new()
	var color = "#ecb879" if index == 4 else "#a1e4ef"
	image.load_svg_from_string('<svg xmlns="http://www.w3.org/2000/svg" width="44" height="44" viewBox="0 0 48 48"><g fill="%s" stroke="%s" stroke-width="2.5" stroke-linejoin="round">%s</g></svg>' % [color, color, shapes[index]])
	return ImageTexture.create_from_image(image)

func open_sheet(index: int) -> void:
	prior_pause = world.paused
	world.paused = true
	refresh_sheets()
	tabs.current_tab = index
	modal.show()

func close_modal() -> void:
	modal.hide()
	world.paused = prior_pause

func toggle_pause() -> void:
	if modal.visible:
		close_modal()
	else:
		super.toggle_pause()

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		var key: int = event.physical_keycode
		if key == KEY_ESCAPE:
			toggle_pause()
			return
		if modal.visible:
			return
		if key == KEY_C or key == KEY_I:
			open_sheet(1 if key == KEY_C else 3)
		elif key == KEY_TAB:
			cycle_mode()
		elif key == KEY_E:
			base_action()
		elif key == KEY_Q:
			world.drink()
		elif key == KEY_SPACE:
			world.dash(movement_input())
		elif key >= KEY_1 and key <= KEY_4:
			var index = key - KEY_1
			world.cast(index, world.target_near_cursor(get_global_mouse_position() / ZOOM, Rules.SKILLS[index].reach))
	if event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT and not modal.visible:
		world.attack(world.target_near_cursor(get_global_mouse_position() / ZOOM, character.stats().reach))

func _process(delta: float) -> void:
	var old: Vector2 = world.player
	world.tick(minf(delta, 0.05), Vector2.ZERO if modal.visible else movement_input())
	player_moving = old.distance_to(world.player) > 0.01
	for enemy in world.enemies:
		moving[enemy.id] = previous_positions.get(enemy.id, enemy.pos).distance_to(enemy.pos) > 0.01
		previous_positions[enemy.id] = enemy.pos
	if not world.paused:
		anim_time += delta
	if world.changed:
		refresh_sheets()
	if not world.messages.is_empty() and world.messages[0] != last_message:
		last_message = world.messages[0]
		toast.text = last_message
		toast_time = 4.0
	if world.cores > last_cores:
		world.effects.append({"kind": "number", "pos": world.player, "text": "+ 晶核", "time": 0.7, "electric": true})
	last_cores = world.cores
	toast_time -= delta
	toast.visible = toast_time > 0
	save_timer -= delta
	if world.save_requested and save_timer <= 0:
		save_game()
		save_timer = 2
	refresh_hud()
	queue_redraw()

func refresh_hud() -> void:
	var stats = character.stats()
	health_text.text = "行者  Lv.%d     %d / %d" % [character.level, ceili(world.hp), int(stats.hp)]
	health_bar.max_value = stats.hp
	health_bar.value = world.hp
	stamina_bar.max_value = stats.stamina
	stamina_bar.value = world.stamina
	load_bar.value = world.load_value
	objective.text = ("外围清理  ·  晶核 %d / 12\n%s" % [world.cores, "返回西侧入口交付  ←" if world.can_deliver() else ("击败节点守卫" if world.boss_spawned else "靠近倒地丧尸取核")]) if world.active else "猎鹰外围 · 安全入口\n整备完成，向街区出发"
	mode_button.text = ("● 自动探索中" if world.mode == 2 else ("◐ 辅助战斗" if world.mode == 1 else "○ 手动操作")) + "  ↻"
	mode_button.tooltip_text = "点击切换手动 / 辅助 / 自动；WASD 随时接管移动"
	growth_button.text = "角色 · 背包" + ("  +%d" % character.attribute_points if character.attribute_points else "")
	base_button.visible = not modal.visible and world.player.distance_to(World.BASE) < 78
	base_button.text = "出发探索  →" if not world.active else ("交付委托  ✓" if world.can_deliver() else "撤回基地  ←")
	base_button.disabled = world.paused
	pause_button.text = "▶" if world.paused else "Ⅱ"
	hint.text = "WASD 移动 · 点击敌人攻击 · 空格闪避" if world.active else "点击门旁「出发探索」开始行动"
	if world.manual_override > 0 and world.mode == 2:
		mode_button.text = "手动接管中  ·  自动待命"
	for i in 4:
		var cd: float = world.skill_cds[i]
		var ratio: float = cd / Rules.SKILLS[i].cd
		cooldowns[i].position = Vector2(0, 70 * (1 - ratio))
		cooldowns[i].size = Vector2(70, 70 * ratio)
		cooldowns[i].visible = cd > 0
		cd_labels[i].text = "%.1f" % cd if cd > 0 else ""
		skill_buttons[i].disabled = world.paused or not world.active or cd > 0 or world.load_value + Rules.SKILLS[i].load > 100

func _draw() -> void:
	if world == null or ground == null:
		return
	draw_set_transform(Vector2.ZERO, 0, Vector2.ONE * ZOOM)
	draw_texture_rect(ground, Rect2(Vector2.ZERO, World.SIZE), false, Color("#a8b8b4"))
	# Solid buildings match the collision footprints, including their front wall.
	for i in world.obstacles.size():
		var r: Rect2 = world.obstacles[i]
		draw_rect(Rect2(r.position + Vector2(10, 12), r.size), Color(0, 0, 0, 0.4))
		draw_rect(r, Color("#333b3b"))
		var roof = Rect2(r.position, r.size - Vector2(0, 20))
		draw_rect(roof, Color("#697474") if i % 2 == 0 else Color("#636967"))
		draw_rect(roof.grow(-5), Color("#475354"))
		for x in range(12, int(r.size.x) - 10, 13):
			draw_line(r.position + Vector2(x, 8), r.position + Vector2(x, r.size.y - 26), Color("#647071"), 2)
		draw_rect(Rect2(r.position + Vector2(17, 20), Vector2(36, 25)), Color("#2e3a3b"))
		draw_circle(r.position + Vector2(35, 32), 9, Color("#7c8a87"))
		for x in range(12, int(r.size.x) - 15, 32):
			draw_rect(Rect2(r.position + Vector2(x, r.size.y - 16), Vector2(19, 9)), Color("#18282b"))
		draw_rect(r, Color("#92968a"), false, 2)
	# Camp gate, beacon and barricades.
	draw_rect(Rect2(5, 227, 90, 146), Color(0.25, 0.65, 0.52, 0.12))
	for y in [211, 369]:
		for x in range(7, 107, 20):
			draw_style_box(panel_style(Color("#737761"), Color("#343e36"), 3), Rect2(x, y, 20, 13))
	for y in [246, 349]:
		draw_rect(Rect2(35, y, 8, 25), Color("#293e3c"))
		draw_circle(Vector2(39, y), 7, Color("#95edbe"))
	draw_arc(World.BASE, 31 + sin(anim_time * 2) * 2, 0, TAU, 40, C_GREEN, 1.5, true)
	text_at("猎鹰入口", Vector2(19, 206), 13, C_GREEN)
	draw_line(Vector2(108, 304), Vector2(147, 304), C_AMBER, 3)
	draw_line(Vector2(147, 304), Vector2(138, 297), C_AMBER, 3)
	draw_line(Vector2(147, 304), Vector2(138, 311), C_AMBER, 3)
	for drop in world.drops:
		var p: Vector2 = drop.pos
		var c: Color = QUALITY_COLORS[int(drop.item.quality)]
		draw_line(p, p + Vector2(0, -35), Color(c, 0.5), 3)
		draw_rect(Rect2(p - Vector2(9, 6), Vector2(18, 12)), Color("#786045"))
		draw_rect(Rect2(p - Vector2(9, 6), Vector2(18, 12)), c, false, 2)
		draw_line(p - Vector2(0, 6), p + Vector2(0, 6), c, 3)
	var actors: Array = world.enemies.duplicate()
	actors.append({"is_player": true, "pos": world.player})
	actors.sort_custom(func(a, b): return a.pos.y < b.pos.y)
	for actor in actors:
		if actor.has("is_player"):
			draw_player()
		else:
			draw_enemy(actor)
	for effect in world.effects:
		draw_effect(effect)
	draw_set_transform(Vector2.ZERO)
	# Small readable backing only where HUD is present.
	draw_style_box(panel_style(Color(0.035, 0.065, 0.07, 0.9), C_LINE), Rect2(18, 15, 300, 80))
	draw_style_box(panel_style(Color(0.035, 0.065, 0.07, 0.85), C_LINE), Rect2(1000, 19, 420, 78))
	if world.paused and not modal.visible:
		draw_rect(Rect2(0, 0, 1440, 900), Color(0, 0, 0, 0.45))
		text_at("已暂停", Vector2(643, 420), 32, C_TEXT)
		text_at("Esc 或右下角继续", Vector2(628, 463), 18, C_DIM)

func actor_sprite(p: Vector2, row: int, height: float, walking: bool, flip: bool, down: bool = false, tint: Color = Color.WHITE) -> void:
	var frame = int(anim_time * 8) % 3 if walking else 0
	var bob = sin(anim_time * 16) * 1.3 if walking else sin(anim_time * 2) * 0.6
	draw_set_transform((p + Vector2(2, 1)) * ZOOM, 0, Vector2(1, 0.32) * ZOOM)
	draw_circle(Vector2.ZERO, height * 0.22, Color(0, 0, 0, 0.4))
	draw_set_transform((p + Vector2(0, bob)) * ZOOM, -PI / 2.8 if down else 0, Vector2(-1 if flip else 1, 1) * ZOOM)
	draw_texture_rect(frames[row][frame], Rect2(-height * 0.4, -height * 0.96, height * 0.8, height), false, tint)
	draw_set_transform(Vector2.ZERO, 0, Vector2.ONE * ZOOM)

func draw_player() -> void:
	var p: Vector2 = world.player
	draw_arc(p, 17, 0, TAU, 36, C_GREEN, 2, true)
	var hurt = false
	var lunge = Vector2.ZERO
	for effect in world.effects:
		if effect.kind == "hurt":
			hurt = true
		elif effect.kind == "slash" or effect.kind == "shot":
			lunge = world.facing * 5
	actor_sprite(p + lunge, 0, 68, player_moving, world.facing.x < 0, false, Color("#ff9999") if hurt else Color.WHITE)
	if world.shield > 0:
		draw_arc(p + Vector2(0, -23), 32, 0, TAU, 48, C_BLUE, 2, true)
	if world.dash_time > 0:
		draw_line(p - world.facing * 35, p, Color(0.5, 0.9, 1, 0.6), 8, true)

func draw_enemy(enemy: Dictionary) -> void:
	var p: Vector2 = enemy.pos
	var height = 65.0 + (enemy.type * 3 if enemy.type < 4 else 35)
	var row = 2 if enemy.type == 1 else (3 if enemy.type >= 2 else 1)
	if enemy.windup > 0:
		draw_circle(enemy.blast, 115, Color(1, 0.15, 0.06, 0.23))
		draw_arc(enemy.blast, 115, 0, TAU, 64, C_RED, 3, true)
		draw_arc(enemy.blast, 115 * (1 - enemy.windup / 1.15), 0, TAU, 64, C_AMBER, 3, true)
		text_at("离开红圈！", enemy.blast + Vector2(-35, -120), 16, C_AMBER)
	var tint = Color("#dcc0ee") if enemy.type == 3 else (Color("#ffb1a1") if enemy.type == 4 else Color.WHITE)
	if enemy.flash > 0:
		tint = Color(2, 2, 2)
	actor_sprite(p, row, height, moving.get(enemy.id, false) and not enemy.down and enemy.stun <= 0, p.x > world.player.x, enemy.down, tint.darkened(0.5) if enemy.down else tint)
	if enemy.down:
		var q = p + Vector2(0, -8 + sin(anim_time * 4) * 3)
		draw_circle(q, 10, Color(0.4, 0.8, 1, 0.12))
		draw_colored_polygon(PackedVector2Array([q + Vector2(0, -7), q + Vector2(4, 0), q + Vector2(0, 7), q - Vector2(4, 0)]), C_BLUE)
		return
	if enemy.hp < enemy.max_hp or enemy.type >= 3:
		draw_rect(Rect2(p + Vector2(-23, -height - 7), Vector2(46, 4)), C_BG)
		draw_rect(Rect2(p + Vector2(-23, -height - 7), Vector2(46 * enemy.hp / enemy.max_hp, 4)), C_RED)
	if enemy.type >= 3:
		text_at(World.ENEMY_TYPES[enemy.type].name, p + Vector2(-28, -height - 12), 12, C_RED)
	if enemy.stun > 0:
		draw_arc(p + Vector2(0, -height), 9, anim_time * 5, anim_time * 5 + PI * 1.5, 16, C_BLUE, 2, true)
