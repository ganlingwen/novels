extends RefCounted

const VERSION = 1
var path: String

func _init(save_path: String = "user://survivor.json") -> void:
	path = save_path

func save_character(character: RefCounted) -> Error:
	var file = FileAccess.open(path + ".tmp", FileAccess.WRITE)
	if file == null:
		return FileAccess.get_open_error()
	file.store_string(JSON.stringify({"version": VERSION, "character": character.snapshot()}, "\t"))
	file.flush()
	var error = file.get_error()
	file.close()
	if error != OK:
		return error
	return DirAccess.rename_absolute(path + ".tmp", path)

func load_character(character: RefCounted) -> String:
	if not FileAccess.file_exists(path):
		return "新角色已就绪。"
	var parser = JSON.new()
	var parse_error = parser.parse(FileAccess.get_file_as_string(path))
	var data = parser.data if parse_error == OK else null
	if data is Dictionary and data.get("version") == VERSION and data.get("character") is Dictionary:
		if character.restore(data.character):
			return "已读取角色进度；从基地继续。"
	var backup = path + ".invalid-" + str(Time.get_unix_time_from_system())
	var error = DirAccess.rename_absolute(path, backup)
	if error != OK:
		return "存档无法读取，也无法备份。请先备份存档目录；本次禁止覆盖。"
	return "存档无效，已保留 .invalid 备份；使用新角色。"
