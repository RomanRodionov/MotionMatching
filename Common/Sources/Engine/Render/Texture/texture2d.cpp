#include "texture2d.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "Engine/Resources/resources.h"

Texture2D::Texture2D()
{
  generateMips = false;
  textureType = TextureType::Texture2DType;
  colorFormat = TextureColorFormat::RGB;
  textureFormat = TextureFormat::Byte;
  wrapping = TextureWrappFormat::Repeat;
  pixelFormat = TexturePixelFormat::Linear;
  textureName = "empty texture";
}
Texture2D::Texture2D(string texture_path_from_textures_folder,
  TextureColorFormat color_format, 
  TextureFormat texture_format, 
  TexturePixelFormat pixel_format,
  TextureWrappFormat wrapping_format, 
  bool generate_mips)
  {
    generateMips = generate_mips;
    textureType = TextureType::Texture2DType;
    colorFormat = color_format;
    textureFormat = texture_format;
    wrapping = wrapping_format;
    pixelFormat = pixel_format;
    
    load(texture_path_from_textures_folder);
  }
  
  void Texture2D::load(const std::string &path)
  {
    textureName = path;
    int w, h, ch;
    stbi_set_flip_vertically_on_load(true);
    auto image = stbi_load(textureName.c_str(), &w, &h, &ch, 0);
		
		if (image)
		{
  
      glGenTextures(1, &textureObject);
      glBindTexture(textureType, textureObject);
			textureWidth = w;
			textureHeight = h;
      
			glTexImage2D(textureType, 0, colorFormat, textureWidth, textureHeight, 0, colorFormat, textureFormat, image);

      glTexParameteri(textureType, GL_TEXTURE_WRAP_S, wrapping);
      glTexParameteri(textureType, GL_TEXTURE_WRAP_T, wrapping);
      if (generateMips)
      {
        glGenerateMipmap(textureType);
        GLenum mipmapMinPixelFormat = pixelFormat == TexturePixelFormat::Linear ? GL_LINEAR_MIPMAP_LINEAR : GL_NEAREST_MIPMAP_NEAREST;
        GLenum mipmapMagPixelFormat = pixelFormat == TexturePixelFormat::Linear ? GL_LINEAR : GL_NEAREST;
        
        glTexParameteri(textureType, GL_TEXTURE_MIN_FILTER, mipmapMinPixelFormat);
        glTexParameteri(textureType, GL_TEXTURE_MAG_FILTER, mipmapMagPixelFormat);
      }
      else
      {
        GLenum minMagixelFormat = pixelFormat == TexturePixelFormat::Linear ? GL_LINEAR : GL_NEAREST;
        glTexParameteri(textureType, GL_TEXTURE_MIN_FILTER, minMagixelFormat);
        glTexParameteri(textureType, GL_TEXTURE_MAG_FILTER, minMagixelFormat);
      }
      glBindTexture(textureType, 0); 
		}
		else
		{
			debug_error("Can't load texture %s!",textureName.c_str());
			return;
		}
  }
TexturePtr make_texture2d()
{
  return make_shared<Texture2D>();
}
TexturePtr make_texture2d(const string& texture_path_from_textures_folder,
  TextureColorFormat color_format, 
  TextureFormat texture_format, 
  TexturePixelFormat pixelFormat,
  TextureWrappFormat wrapping, 
  bool generate_mips)
{
  return make_shared<Texture2D>(texture_path_from_textures_folder, color_format, texture_format, pixelFormat, wrapping, generate_mips);
}
TexturePtr make_texture2d(const string& texture_path_from_textures_folder,
  TexturePixelFormat pixelFormat,
  TextureWrappFormat wrapping, 
  bool generate_mips,
  TextureColorFormat color_format, 
  TextureFormat texture_format)
{
  return make_shared<Texture2D>(texture_path_from_textures_folder, color_format, texture_format, pixelFormat, wrapping, generate_mips);
}

void Texture2D::free()
{

}
bool Texture2D::edit()
{
  return false;
}

ResourceRegister<Texture2D> texture2DRegister({".jpg", ".png"});